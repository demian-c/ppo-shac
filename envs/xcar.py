import os
import sys
import gym
import torch
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv

class EightDriftEnv(GPUVectorizedCarEnv):
    def __init__(self, preset_name, n, **kwargs):
        super().__init__(preset_name, n, **kwargs)
        self.max_steps = 2000
        self.num_states = 36
        self.num_actions = 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.action_space = gym.spaces.Box(low=np.array([-0.6, 1.]), high=np.array([0.6, 7.]), shape=(2,))
        self.state_space = self.observation_space
        dummy_obs = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_obs = [dummy_obs, dummy_obs]  # most recent first

        # Initialize waypoints
        num_waypoints = 2400      # one waypoint per 0.005m
        waypoint_x = torch.sin(torch.arange(num_waypoints) / num_waypoints * 4 * np.pi)
        waypoint_y = torch.cat((
            1 - torch.cos(torch.arange(num_waypoints / 2) / num_waypoints * 4 * np.pi),
            -1 + torch.cos(torch.arange(num_waypoints / 2) / num_waypoints * 4 * np.pi),
        ))
        waypoint_veldir = torch.cat((
            torch.arange(num_waypoints / 2) / num_waypoints * 2,
            num_waypoints / 2 - torch.arange(num_waypoints / 2) / num_waypoints * 2,
        )) * 2 * np.pi
        # wrap to (-pi, pi)
        waypoint_veldir = torch.atan2(torch.sin(waypoint_veldir), torch.cos(waypoint_veldir))
        waypoint_betaref = torch.cat((
            -torch.arange(num_waypoints / 8) / num_waypoints * 8,
            -torch.ones((num_waypoints // 4, )),
            torch.arange(num_waypoints / 4) / num_waypoints * 8 - 1,
            torch.ones((num_waypoints // 4, )),
            1 - torch.arange(num_waypoints / 8) / num_waypoints * 8,
        ))
        self.waypoints = torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_betaref), dim=1).to(self.device)

        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)   # 0 = not done, 1 = failed, 2 = succeeded
        self.no_progress_count = torch.zeros(self.n, dtype=torch.uint8, device=self.device)

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def find_nearest_waypoint(self, x, y, veldir, beta):
        # Look ahead 20 steps
        ind = torch.stack([self.progress + i for i in range(20)], dim=1) % len(self.waypoints)
        waypoints_filtered = self.waypoints[ind]
        current = torch.stack((x, y, veldir, beta), dim=1)
        diff = current.unsqueeze(1) - waypoints_filtered
        # wrap angle diff to (-pi, pi)
        angle_diff = torch.atan2(torch.sin(diff[:, :, 2]), torch.cos(diff[:, :, 2]))
        pos_diff = torch.hypot(diff[:, :, 0], diff[:, :, 1] + 0.01)
        beta_diff = diff[:, :, 3]
        # total_diff = torch.abs(angle_diff) + pos_diff
        self.step_progress = torch.argmin(pos_diff, 1)
        self.progress = (self.progress + self.step_progress) % len(self.waypoints)
        self.no_progress_count[self.step_progress != 0] = 0
        self.no_progress_count[self.step_progress == 0] += 1

        # u = vector from nearest waypoint to current pos; v = velocity direction at nearest waypoint
        # theta = angle from u to v
        # theta > 0 means current pos is to the right of waypoint
        best = self.step_progress.unsqueeze(1)
        ux = diff[:, :, 0].gather(1, best).squeeze(1)
        uy = diff[:, :, 1].gather(1, best).squeeze(1)
        u_angle = torch.atan2(uy, ux + 0.01)
        v_angle = waypoints_filtered[:, :, 2].gather(1, best).squeeze(1)
        theta = v_angle - u_angle
        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        pos_diff_sign = torch.sign(theta)
        self.angle_diff = angle_diff.gather(1, best).squeeze(1)
        self.signed_pos_diff = pos_diff_sign * pos_diff.gather(1, best).squeeze(1)
        self.beta_diff = beta_diff.gather(1, best).squeeze(1)

    def obs(self):
        x = self.s[:, 0]
        y = self.s[:, 1]
        psi = self.s[:, 2]
        c_psi = torch.cos(psi)
        s_psi = torch.sin(psi)
        r = self.es[:, 2]
        beta = self.es[:, 7]
        v = self.es[:, 6]
        vfx = self.es[:, 8]
        vfy = self.es[:, 9]
        vrx = self.es[:, 10]
        vry = self.es[:, 11]
        sfx = torch.clamp(self.es[:, 12], -2., 2.)
        sfy = torch.clamp(self.es[:, 13], -2., 2.)
        srx = torch.clamp(self.es[:, 14], -2., 2.)
        sry = torch.clamp(self.es[:, 15], -2., 2.)
        last_delta = self.u[:, 0]
        last_omega = self.u[:, 1]

        xd = self.s[:, 3]
        yd = self.s[:, 4]
        veldir = torch.atan2(yd, xd + 0.01)
        self.find_nearest_waypoint(x, y, veldir, beta)
        angle_diff = self.angle_diff
        signed_pos_diff = self.signed_pos_diff
        beta_diff = self.beta_diff

        # look ahead 3m
        la = [self.waypoints[(self.progress + i) % len(self.waypoints)] for i in (200, 400, 600)]
        def convert(waypoints):
            """Cast into current reference frame"""
            x1 = waypoints[:, 0]
            y1 = waypoints[:, 1]
            veldir1 = waypoints[:, 2]
            beta1 = waypoints[:, 3]
            theta = torch.atan2(y1 - y, x1 - x) - veldir
            l = torch.hypot(y1 - y, x1 - x)
            x2 = l * torch.cos(theta)
            y2 = l * torch.sin(theta)
            veldir2 = veldir1 - veldir
            cv2 = torch.cos(veldir2)
            sv2 = torch.sin(veldir2)
            beta2 = beta1 - beta
            x2, y2, cv2, sv2, beta2 = map(lambda t: torch.unsqueeze(t, 1), [x2, y2, cv2, sv2, beta2])
            return torch.cat([x2, y2, cv2, sv2, beta2], 1)
        la_converted = [convert(waypoints) for waypoints in la]

        failed = (self.no_progress_count >= 20) | (torch.abs(angle_diff) > 1.) | (torch.abs(signed_pos_diff) > 1.)
        succeeded = (self.step_count >= self.max_steps)
        self.is_done[failed] = 1
        self.is_done[succeeded] = 2

        x, y, c_psi, s_psi, r, beta, v, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, last_delta, last_omega, angle_diff, signed_pos_diff, beta_diff, prog = map(lambda t: torch.unsqueeze(t, 1), [x, y, c_psi, s_psi, r, beta, v, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, last_delta, last_omega, angle_diff, signed_pos_diff, beta_diff, self.step_progress])
        obs = torch.cat([x, y, c_psi, s_psi, r, beta, v, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, last_delta, last_omega, angle_diff, signed_pos_diff, beta_diff, prog] + la_converted, 1)
        return obs

    def reward(self):
        rew_pos = - self.signed_pos_diff ** 2
        rew_dir = - self.angle_diff ** 2
        rew_prog = torch.clamp(self.step_progress, 0.0, 5.0)
        obs = self.recent_obs[0]
        last_obs = self.recent_obs[1]
        delta = obs[:, 15]
        omega = obs[:, 16]
        last_delta = last_obs[:, 15]
        last_omega = last_obs[:, 16]
        rew_smooth = - ((delta - last_delta) ** 2 + 1e-2 * (omega - last_omega) ** 2)
        rew_beta = - self.beta_diff ** 2
        v = obs[:, 6]
        rew_lowspeed = torch.clamp(v, 0., 0.5) - 0.5

        rew = 2 * rew_pos + 0.5 * rew_dir + 0.2 * rew_prog + 0.02 * rew_smooth + 2 * rew_beta + 0.1 * rew_lowspeed

        print(rew_pos.mean().item(), rew_dir.mean().item(), rew_prog.mean().item(), rew_smooth.mean().item(), rew_beta.mean().item(), rew.mean().item())
        return rew

    def done(self):
        return self.is_done

    def reset(self):
        super().reset()
        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.no_progress_count = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        return self.obs()

    def info(self):
        return {
            "time_outs": (self.is_done == 2)
        }

    def reset_done_envs(self):
        """Only reset envs that are already done."""
        is_done = self.is_done.bool()
        size = torch.sum(is_done)
        self.step_count[is_done] = 0
        self.progress = self.progress.clone()
        self.progress[is_done] = torch.randint(self.waypoints.shape[0], (size,), device=self.device)
        self.step_progress = self.step_progress.clone()
        self.step_progress[is_done] = 0
        self.no_progress_count = self.no_progress_count.clone()
        self.no_progress_count[is_done] = 0
        def gen_random_state(prog):
            x = self.waypoints[prog, 0]
            y = self.waypoints[prog, 1]
            veldir = self.waypoints[prog, 2]
            beta = self.waypoints[prog, 3]
            v = torch.rand(prog.shape, device=self.device) * 3
            xd = v * torch.cos(veldir)
            yd = v * torch.sin(veldir)
            psi = veldir - beta
            psid = -torch.sign(beta) * (1 + torch.rand(prog.shape, device=self.device) * 2)
            x, y, psi, xd, yd, psid = map(lambda t: torch.unsqueeze(t, 1), [x, y, psi, xd, yd, psid])
            return torch.cat([x, y, psi, xd, yd, psid], 1)
        self.s = self.s.clone()
        self.s[is_done, :] = gen_random_state(self.progress[is_done])
        self.u = self.u.clone()
        self.u[is_done, :] = 0
        self.is_done[:] = 0

    def step(self, action, **kwargs):
        self.reset_done_envs()
        obs, reward, done, info = super().step(action, **kwargs)
        self.update_recent_obs(obs)
        return obs, reward, done, info

    def detach(self):
        super().detach()
        for i in range(2):
            if self.recent_obs[i] is not None:
                self.recent_obs[i] = self.recent_obs[i].detach()
