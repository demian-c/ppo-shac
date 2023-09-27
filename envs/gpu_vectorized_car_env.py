import numpy as np
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import yaml
import gym
import time
import random
import csv
from pathlib import Path


class CarDynamics(nn.Module):
    def __init__(self, u, p_body, p_tyre, differentiable=False, disturbance=None):
        """
        u is batch_size * 3, corresponding to (delta, omegaf, omegar).
        """
        super().__init__()
        to_param = nn.Parameter if differentiable else lambda x: x
        self.u = to_param(u)
        self.p_body = to_param(p_body)
        self.p_tyre = to_param(p_tyre)
        self.disturbance = disturbance

    def compute_extended_state(self, s):
        """
        s is batch_size * 6, corresponding to (x, y, psi, xd, yd, psid).
        """
        # Decoding
        x = s[:, 0]
        y = s[:, 1]
        psi = s[:, 2]
        xd = s[:, 3]
        yd = s[:, 4]
        psid = s[:, 5]
        delta = self.u[:, 0]
        omegaf = self.u[:, 1]
        omegar = self.u[:, 2]
        lf = self.p_body[:, 0]
        lr = self.p_body[:, 1]
        m = self.p_body[:, 2]
        h = self.p_body[:, 3]
        g = self.p_body[:, 4]
        Iz = self.p_body[:, 5]
        B = self.p_tyre[:, 0]
        C = self.p_tyre[:, 1]
        D = self.p_tyre[:, 2]
        E = self.p_tyre[:, 3]

        eps = 1e-3
        v = torch.hypot(xd, yd + eps)
        beta = torch.atan2(yd, xd + eps) - psi
        # wrap beta to (-pi, pi)
        beta = torch.atan2(torch.sin(beta), torch.cos(beta))
        vfx = v * torch.cos(beta - delta) + psid * lf * torch.sin(delta)
        vfy = v * torch.sin(beta - delta) + psid * lf * torch.cos(delta)
        vrx = v * torch.cos(beta)
        vry = v * torch.sin(beta) - psid * lr
        sfx = (vfx - omegaf) / (omegaf + eps)
        sfy = (vfy) / (omegaf + eps)
        srx = (vrx - omegar) / (omegar + eps)
        sry = (vry) / (omegar + eps)
        sf = torch.hypot(sfx, sfy + eps)
        sr = torch.hypot(srx, sry + eps)
        pacejka = lambda slip: D * torch.sin(C * torch.atan(B * slip - E * (B * slip - torch.atan(B * slip))))
        muf = pacejka(sf)
        mur = pacejka(sr)
        alphaf = torch.atan2(sfy, sfx + eps)
        alphar = torch.atan2(sry, srx + eps)
        mufx = -torch.cos(alphaf) * muf
        mufy = -torch.sin(alphaf) * muf
        murx = -torch.cos(alphar) * mur
        mury = -torch.sin(alphar) * mur
        G = m * g
        l = lf + lr
        ffz = (lr * G - h * G * murx) / (l + h * (mufx * torch.cos(delta) - mufy * torch.sin(delta) - murx))
        frz = G - ffz
        ffx = mufx * ffz
        ffy = mufy * ffz
        frx = murx * frz
        fry = mury * frz
        if self.disturbance is not None:
            ffx += self.disturbance[:, 0]
            ffy += self.disturbance[:, 1]
            frx += self.disturbance[:, 2]
            fry += self.disturbance[:, 3]
        
        xdd = 1 / m * (ffx * torch.cos(psi + delta) - ffy * torch.sin(psi + delta) + frx * torch.cos(psi) - fry * torch.sin(psi))
        ydd = 1 / m * (ffx * torch.sin(psi + delta) + ffy * torch.cos(psi + delta) + frx * torch.sin(psi) + fry * torch.cos(psi))
        psidd = 1 / Iz * ((ffy * torch.cos(delta) + ffx * torch.sin(delta)) * lf - fry * lr)

        # cast from shape [batch_size,] to [batch_size, 1]
        xd, yd, psid, xdd, ydd, psidd, v, beta, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, sf, sr, muf, mur, alphaf, alphar, mufx, mufy, murx, mury, ffz, frz, ffx, ffy, frx, fry = map(lambda t: torch.unsqueeze(t, 1), [xd, yd, psid, xdd, ydd, psidd, v, beta, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, sf, sr, muf, mur, alphaf, alphar, mufx, mufy, murx, mury, ffz, frz, ffx, ffy, frx, fry])

        return torch.cat([xd, yd, psid, xdd, ydd, psidd, v, beta, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, sf, sr, muf, mur, alphaf, alphar, mufx, mufy, murx, mury, ffz, frz, ffx, ffy, frx, fry], 1)

    def forward(self, t, s):
        es = self.compute_extended_state(s)
        return es[:, :6]


class GPUVectorizedCarEnv:
    def __init__(self,
        preset_name,
        n,
        dt=0.01,
        solver="euler",
        device="cuda:0",
        drivetrain="4wd",
        disturbance_param=None,
        randomize_param={},
        random_seed=None,
        **kwargs,
    ):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.num_states = 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.state_space = self.observation_space
        self.drivetrain = drivetrain
        if drivetrain == "4wd":
            self.num_actions = 2
            self.action_space = gym.spaces.Box(low=np.array([-0.6, 0.]), high=np.array([0.6, 7.]), shape=(2,))
            self.cast_action = lambda u: torch.cat(list(map(lambda v: torch.unsqueeze(v, 1), [u[:, 0], u[:, 1], u[:, 1]])), 1)
        elif drivetrain == "iwd":
            self.num_actions = 3
            self.action_space = gym.spaces.Box(low=np.array([-0.6, 0., 0.]), high=np.array([0.6, 7., 7.]), shape=(3,))
            self.cast_action = lambda u: u

        self.preset_name = preset_name
        self.n = n
        self.dt = dt
        self.solver = solver
        self.device = torch.device(device)
        file_path = os.path.dirname(__file__)
        with open(os.path.join(file_path, "presets.yaml")) as f:
            presets = yaml.safe_load(f)
            params = presets[preset_name]["parameters"]
        self.p_body = torch.zeros((n, 6), device=self.device)
        self.p_body[:, 0] = params["lF"]
        self.p_body[:, 1] = params["lR"]
        self.p_body[:, 2] = params["m"]
        self.p_body[:, 3] = params["h"]
        self.p_body[:, 4] = params["g"]
        self.p_body[:, 5] = params["Iz"]
        self.p_tyre = torch.zeros((n, 4), device=self.device)
        self.p_tyre[:, 0] = params["B"]
        self.p_tyre[:, 1] = params["C"]
        self.p_tyre[:, 2] = params["D"]
        self.p_tyre[:, 3] = params["E"]
        self.randomize_param = randomize_param
        self.s = torch.zeros((n, 6), device=self.device)
        self.dynamics = None
        self.disturbance_param = disturbance_param
        if disturbance_param is not None:
            self.disturbance = torch.zeros((self.n, 4), device=self.device)
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.total_step_count = 0
        self.saved_data = []

    def randomize_item_(self, env_mask, override, key, target):
        num = int(torch.sum(env_mask).item())
        if key in override:
            target[env_mask] = override[key]
        elif key in self.randomize_param:
            lo, hi = self.randomize_param[key]
            target[env_mask] = lo + (hi - lo) * torch.rand(num, device=self.device)

    def randomize_items_(self, env_mask, override):
        rand_item_ = lambda key, target: self.randomize_item_(env_mask, override, key, target)
        rand_item_("B", self.p_tyre[:, 0])
        rand_item_("C", self.p_tyre[:, 1])
        rand_item_("D", self.p_tyre[:, 2])

    def randomize(self, env_mask=None, override={}):
        if env_mask is None:
            env_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.randomize_items_(env_mask, override)

    def obs(self):
        return self.s

    def reward(self):
        return torch.zeros(self.n, device=self.device)

    def done(self):
        return torch.zeros(self.n, device=self.device)

    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def disturbed_dynamics(self):
        a, w = self.disturbance_param
        self.disturbance = a * self.disturbance + w * torch.randn((self.n, 4), device=self.device)
        return CarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre, disturbance=self.disturbance)

    def reset(self):
        self.randomize()
        self.s = torch.zeros((self.n, 6), device=self.device)
        self.u = torch.zeros((self.n, self.num_actions), device=self.device)
        self.dynamics = CarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre)
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.total_step_count = 0
        self.saved_data = []
        return self.obs()

    def step(self, u, override_s=None):
        self.u = u
        self.dynamics = CarDynamics(self.cast_action(u), self.p_body, self.p_tyre) if self.disturbance_param is None else self.disturbed_dynamics()
        if override_s is None:
            self.s = odeint(self.dynamics, self.s, torch.tensor([0., self.dt]), method=self.solver)[1, :, :]
        else:
            self.s[:] = torch.tensor(override_s).unsqueeze(0)
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count += 1
        self.total_step_count += 1
        obs, reward, done, info = self.obs(), self.reward(), self.done(), self.info()
        if torch.all(done) and self.saved_data:
            basename = time.strftime("%Y%m%d-%H%M%S")
            Path("data").mkdir(parents=True, exist_ok=True)
            with open(os.path.join("data", basename + ".csv"), 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for i in range(len(self.saved_data)):
                    s = self.saved_data[i][0]
                    x, y, psi = s[0].item(), s[1].item(), s[2].item()
                    writer.writerow([x, y, psi])
            torch.save(self.saved_data, os.path.join("data", basename + ".pth"))
            print("Total steps:", self.total_step_count)
            exit(0)
        return obs, reward, done, info

    def render(self, **kwargs):
        """Save rollout data to emulate rendering."""
        self.saved_data.append((self.s[0, :].cpu(), self.u[0, :].cpu(), self.es[0, :].cpu()))

    def detach(self):
        """Clear the gradient stored in the current state of the environment."""
        self.s = self.s.detach()
        self.u = self.u.detach()
        self.es = self.es.detach()


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import time

    # Compare solvers
    env_euler = GPUVectorizedCarEnv("racecar", 1, solver="euler", differentiable=False)
    env_rk = GPUVectorizedCarEnv("racecar", 1, solver="dopri5", differentiable=False)
    env_rk8 = GPUVectorizedCarEnv("racecar", 1, solver="dopri8", differentiable=False)

    traj_euler = [env_euler.reset().cpu().numpy()]
    traj_rk = [env_rk.reset().cpu().numpy()]
    # traj_rk8 = [env_rk8.reset().cpu().numpy()]
    
    for i in range(600):
        if i < 100:
            u = [0., 1.]
        elif 100 <= i < 200:
            u = [0., 4.]
        elif 200 <= i < 300:
            u = [0., 1.]
        elif 300 <= i < 400:
            u = [0.4, 4.]
        else:
            u = [-0.4, 4.]

        s_euler, _, _, _ = env_euler.step(torch.tensor([u], device=torch.device("cuda:0")))
        s_rk, _, _, _ = env_rk.step(torch.tensor([u], device=torch.device("cuda:0")))
        # s_rk8, _, _, _ = env_rk8.step(torch.tensor([u], device=torch.device("cuda:0")))

        traj_euler.append(s_euler.detach().cpu().numpy())
        traj_rk.append(s_rk.detach().cpu().numpy())
        # traj_rk8.append(s_rk8.cpu().numpy())

    plt.figure(dpi=300)
    plt.plot([s[0][0] for s in traj_euler], [s[0][1] for s in traj_euler], label="Euler")
    plt.plot([s[0][0] for s in traj_rk], [s[0][1] for s in traj_rk], label="RK5")
    # plt.plot([s[0][0] for s in traj_rk8], [s[0][1] for s in traj_rk8], label="RK8")
    plt.legend()
    plt.axis("equal")

    # Test large-scale parallelization
    ns = [10 ** i for i in range(7)]
    def measure_time(n, solver, device):
        env = GPUVectorizedCarEnv("racecar", n, solver=solver, device=device)
        u = torch.tensor([[0.1, 10.] for _ in range(n)], device=torch.device(device))
        start_time = time.time()
        for i in tqdm(range(100)):
            env.step(u)
        elapsed = time.time() - start_time
        return elapsed
    times_euler_gpu = [measure_time(n, "euler", "cuda:0") for n in ns]
    times_rk_gpu = [measure_time(n, "dopri5", "cuda:0") for n in ns]
    times_euler_cpu = [measure_time(n, "euler", "cpu") for n in ns]
    times_rk_cpu = [measure_time(n, "dopri5", "cpu") for n in ns]


    plt.figure(dpi=300)
    plt.loglog(ns, times_euler_gpu, label="Euler (GPU)")
    plt.loglog(ns, times_rk_gpu, label="RK5 (GPU)")
    plt.loglog(ns, times_euler_cpu, label="Euler (64-core CPU)")
    plt.loglog(ns, times_rk_cpu, label="RK5 (64-core CPU)")
    plt.legend()
    plt.xlabel("# of instances")
    plt.ylabel("Time of performing 1s simulation")
