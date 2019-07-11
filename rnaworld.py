import gym
from gym import spaces
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import pyglet
import os
import time
import pymol
pymol.finish_launching(['pymol', '-qc'])
cmd = pymol.cmd

class RNAWorld2D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        with open('data/some_rna.pkl','rb') as f:
            self.dataset = pickle.load(f)

        self.action_space = None
        self.pseudoknot_space = None
        self.state = None
        self.info = None
        self.viewer = None
        self.action_empty = False
        self.pseudo_mode = False

        self.set_rna(np.random.choice(self.dataset))

    def seed(self, seed):
        np.random.seed(seed=seed)

    def random_rna(self):
        self.set_rna(np.random.choice(self.dataset))

    def set_rna(self, rna):
        L = rna['len']
        self.info = rna
        self.info['pairs'] = [(a-1, b-1) for a,b in rna['pairs']]

    def step(self, action):
        L = self.info['len']
        i, j = action
        self.state[i, j] = 1
        self.state[j, i] = 1
        self.info['pred_pairs'].append((i, j))

        sec_list = list(self.info['pred_sec'])
        sec_list[i] = '(' if not self.pseudo_mode else '['
        sec_list[j] = ')' if not self.pseudo_mode else ']'
        self.info['pred_sec'] = ''.join(sec_list)

        if not self.action_empty:
            # update action space
            remain = []
            for a, b in self.action_space:
                if b < i:
                    remain.append((a, b))
                elif a > j:
                    remain.append((a, b))
                elif a < i and j < b:
                    remain.append((a, b))
                elif i < a and b < j:
                    remain.append((a, b))
            self.action_space = remain

            if len(remain) == 0:
                self.action_empty = True

            # update pseudoknot space
            remain = []
            for a, b in self.pseudoknot_space:
                if a != i and a != j and b != i and b != j:
                    remain.append((a, b))
            self.pseudoknot_space = remain
        else:
            remain = []
            for a, b in self.pseudoknot_space:
                if b < i:
                    remain.append((a, b))
                elif a > j:
                    remain.append((a, b))
                elif a < i and j < b:
                    remain.append((a, b))
                elif i < a and b < j:
                    remain.append((a, b))
            self.pseudoknot_space = remain

        # calculate reward
        # r = tpr * (1 - fpr)
        groud = set(self.info['pairs'])
        pred = set(self.info['pred_pairs'])
        tpr = len(groud & pred) / len(groud)
        r_mcc = tpr * (len(groud & pred) / len(pred))
        # 1 - normed hamming distance
        r_hamming = sum([self.info['pred_sec'][i] == self.info['sec'][i] for i in range(L)]) / L

        reward = r_mcc + r_hamming

        if self.pseudo_mode:
            done = (2 - reward) < 0.001 or (self.action_empty and len(self.pseudoknot_space) == 0)
        else:
            done = (2 - reward) < 0.001 or self.action_empty

        return self.state, reward, done, self.info

    def reset(self):
        self.action_empty = False
        L = self.info['len']
        seq = self.info['seq']
        self.info['pred_sec'] = '.' * L
        self.info['pred_pairs'] = []

        adj = np.zeros((L, L))
        for i in range(L-1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1

        self.state = adj

        node = np.zeros((L, 8))
        m = {'A':[0,6],'U':[2,4],'G':[1,7],'C':[3,5]}
        for i in range(L):
            node[i, m[seq[i]]] = 1

        self.action_space = []
        self.pseudoknot_space = []
        for i in range(L):
            for j in range(i+4, L):
                a, b = seq[i], seq[j]
                if sorted((a, b)) in [['A', 'U'], ['C', 'G'], ['G', 'U']]:
                    self.action_space.append((i, j))
                    self.pseudoknot_space.append((i, j))

        return self.state, node, self.info

    def render(self, mode='human'):
        fig = plt.figure(20, figsize=(12, 8))
        fig.clf()
        g_nx = nx.from_numpy_matrix(self.state)
        opts = self._make_opts(self.info['seq'], g_nx)
        nx.draw_kamada_kawai(g_nx, **opts)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape((h, w, 3))

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer(maxwidth=w)
        self.viewer.imshow(data)
        return self.viewer.isopen

    def _make_opts(self, seq, g):
        ncolor_map = {'A':'#EA4335', 'U':'#34A853', 'C':'#4285F4', 'G':'#FBBC05'}
        ecolor_map = ['#4285F4', '#ED7D31', '#34A853']
        ncolor = [ncolor_map[a] for a in seq]
        ecolor = [ecolor_map[0] if abs(a-b) == 1 else ecolor_map[1] for (a, b) in g.edges]

        opts = {
            'with_labels': True,
            'font_weight': 'bold',
            'node_color': ncolor,
            'node_size': 300,
            'font_size': 8,
            'font_color': 'white',
            'edge_color': ecolor
        }

        return opts

    def close(self):
        return None

class RNAWorld3D(gym.Env):
    def __init__(self):
        super().__init__()

        self.info = None
        self.state = None
        self.viewer = None
        self.atom_coords = None
        self.action_space = None
        self.observation_spce = None # spaces.Box(-high, high)
        self.WINDOW_W = 1000
        self.WINDOW_H = 800
        self.snap_dir = ''
        self.snap_count = 0

    def seed(self, seed):
        np.random.seed(seed=seed)

    def set_rna(self, pdb_file):
        cmd.load(pdb_file)
        rid = cmd.get_names()[0]
        seq = ''.join(cmd.get_fastastr().split('\n')[1:])
        L = len(seq)
        cmd.remove('hetatm')

        cmd.show_as('cartoon')
        cmd.zoom()

        self.info = {'id': rid, 'seq': seq, 'len': L}
        self.atom_coords = cmd.get_coords()
        self.snap_dir = f'{rid}-3D'

        if not os.path.exists(self.snap_dir):
            os.mkdir(self.snap_dir)
        else:
            for f in os.listdir(self.snap_dir):
                os.remove(self.snap_dir+'/'+f)

        # only use one chain
        # chain = cmd.get_chains()[0]
        # model = cmd.get_model(f'chain {chain}')

    def reset(self):
        L = self.info['len']
        seq = self.info['seq']

        node = np.zeros((L, 8))
        m = {'A':[0,6],'U':[2,4],'G':[1,7],'C':[3,5]}
        for i in range(L):
            node[i, m[seq[i]]] = 1

        self.state = self.random_init_atom_coords()

        return self.state, node, self.info

    def random_init_atom_coords(self):
        return NotImplemented

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.WINDOW_W, self.WINDOW_H)

        img_file = f'{self.snap_dir}/{self.snap_count}.png'
        cmd.png(img_file, self.WINDOW_W, self.WINDOW_H, dpi=150, ray=1)
        self.snap_count += 1
        
        while not os.path.exists(img_file):
            time.sleep(0.5)

        win = self.viewer.window
        img = pyglet.image.load(img_file)
        win.dispatch_events()
        win.clear()
        img.blit(0,0)
        win.flip()
        return self.viewer.isopen

    def step(self, action):
        return NotImplemented

    def close(self):
        return None

def test_RNAWorld2D():
    env = RNAWorld2D()
    env.seed(2019)
    env.random_rna()
    ob, node, info = env.reset()
    env.render()

    print(f'info:\n{info}\nob:\n{ob}\n')

    while True:
        if len(env.action_space) > 0:
            idx = np.random.randint(len(env.action_space))
            action = env.action_space[idx]
        else:
            idx = np.random.randint(len(env.pseudoknot_space))
            action = env.pseudoknot_space[idx]

        ob, reward, done, info = env.step(action)
        env.render()

        print(f"action: {action}\nreward: {reward}\ndone: {done}\nsec:\n{info['sec']}\npred sec:\n{info['pred_sec']}\n")

        if done:
            input()
            break
    env.close()

def test_RNAWorld3D():
    env = RNAWorld3D()
    env.set_rna('data/1y26.cif')
    env.render()

    ob, node, info = env.reset()
    env.render()

if __name__ == '__main__':
    test_RNAWorld2D()
