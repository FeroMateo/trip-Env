import numpy as np
import gym
import mujoco_py

from typing import Optional


"""
Descripcion:
    Este entorno corresponde al problema 'R0-BB1' definido por Mateo Ruiz


    #Action Space
    Esta definido por un 'ndarray' con forma '(6,)' donde pueden tomar valores
    de '{0,1,2,3,4,5}' indicando la direccion hacia donde el agente se mueve.

    | Num | Action                  |
    |-----|-------------------------|
    | 0   | Moverse adelante        |
    | 1   | Moverse adelante/izquierda|
    | 2   | Moverse izquierda       |
    | 3   | Moverse adelante/derecha|
    | 4   | Moverse derecha         |
    | 5   | Moverse atras           |


    #Observation Space

    Esta definido por un 'ndarray' con forma '(6,)' correspondientes a:

    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Agent Velocity        | -Inf                 | Inf                |
    | 1   | Distance Sensor       | 0                    | 150                |
    | 2   | Distance Sensor       | 0                    | 150                |
    | 3   | Distance Sensor       | 0                    | 150                |
    | 4   | Distance Sensor(LiDAR)| 0                    | 150                |
    | 5   | Distance Sensor(LiDAR)| 0                    | 150                |

    #Rewards

    Como el objetivo es mantener al agente sin chocarse lo maximo posible, se
    lo recompensara con +1 por cada step tomado, incluso el de terminacion.

    #Episode Termination

    El episodio termina si una de las siguientes ocurre:
        1. Algun sensor de distancia tiene valor 0
        2. La duracion supera los 500 puntos

    #Arguments

"""



class TripEnv(gym.Env):
    def __init__(self):

        self.state = None

        self.model_path = "/Users/mateoruiz/Desktop/Robot2/Modelo3d/trip_model.xml"
        self.frame_skip = 800
        self.model = mujoco_py.load_model_from_path(self.model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

        self.viewer = self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewers = {}
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.action_space = gym.spaces.Discrete(6)

        high = np.array([150,150,150])
        low = np.array([0,0,0])

        self.observation_space = gym.spaces.Box(low,high,dtype=np.float64)

    def _get_obs(self):

        """
        En teoria recibo los valores input de los sensores y los redondeo a un numero redondo
        """
        data = list(self.sim.data.sensordata)

        return np.array(data)


    def step(self,action):

        self.state = self._get_obs()

        sensors= self.state

        done = bool(
            round(min(sensors))==0
        )

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        info = {}

        self.do_simulation(self.frame_skip,action)

        return self.state, reward, done, info


    def set_state(self, qpos, qvel):

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        self.set_state(0,0)

        return self._get_obs()

    def render(
        self,
        mode="human",
        camera_id=None,
        camera_name=None,
    ):
        
        if camera_id is not None and camera_name is not None:
            raise ValueError(
                "Both `camera_id` and `camera_name` cannot be"
                " specified at the same time.")
        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = "track"
        self.viewer.render()


    def do_simulation(self,n_frames,action):

        if action == 0:
             valores = [0.1,0.1,0]
        elif action == 1:
            valores = [0.1 ,0.1,-0.5]
        elif action == 2:
            valores = [0.2,0.2,-1]
        elif action == 3:
            valores = [0.1,0.1,0.5]
        elif action == 4:
            valores = [0.2,0.2,1]
        elif action == 5:
            valores = [-0.3,-0.3,0]


        #RUEDA DERECHA vista desde atras
        self.sim.data.ctrl[0] = valores[0]
        #RUEDA IZQUIEDA vista desde atras
        self.sim.data.ctrl[1] = valores[1]

        self.sim.data.ctrl[2] = valores[2]

        for _ in range(n_frames):
            self.sim.step()
