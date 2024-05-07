import logging
import numpy as np
from collections import namedtuple
import torch
from PIL import Image
import torchvision.transforms.functional as TF
TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#print('device = ', device)
RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 
def extract_featuresV2(pstate, soccer_state, opponent_state, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of score-line 
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)

    return features 

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, ActionNet):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'imitationevens.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

class ActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm2d(6),
            torch.nn.Conv2d(6, 16, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 5, stride=2),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(32, 6)
    
    def forward(self, x):
        f = self.network(x)
        return self.classifier(f.mean(dim=(2,3)))

class Actor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
    
    def __call__(self, image, **kwargs):
        output = self.action_net(TF.to_tensor(image)[None])[0]

        action = pystk.Action()
        action.acceleration = output[0]
        action.steer = output[2]
        action.brake= output[3]
        return action


def noisy_actor(actor, noise_std=5):
    def act(**kwargs):
        action = actor(**kwargs)
        action.steer += np.random.normal(0, noise_std)
        return action
    return act


action_net = ActionNet()
actor = Actor(action_net)

def to_native(o):
    # Super obnoxious way to hide pystk
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)


class AIRunner:
    agent_type = 'state'
    is_ai = True

    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    def info(self):
        return RunnerInfo('state', None, 0)


class TeamRunner:
    agent_type = 'state'
    _error = None
    _total_act_time = 0

    def __init__(self, team_or_dir):
        from pathlib import Path
        try:
            from grader import grader
        except ImportError:
            try:
                from . import grader
            except ImportError:
                import grader

        self._error = None
        self._team = None
        try:
            if isinstance(team_or_dir, (str, Path)):
                assignment = grader.load_assignment(team_or_dir)
                if assignment is None:
                    self._error = 'Failed to load submission.'
                else:
                    self._team = assignment.Team()
            else:
                self._team = team_or_dir
        except Exception as e:
            self._error = 'Failed to load submission: {}'.format(str(e))
        if hasattr(self, '_team') and self._team is not None:
            self.agent_type = self._team.agent_type

    def new_match(self, team: int, num_players: int) -> list:
        self._total_act_time = 0
        self._error = None
        try:
            r = self._team.new_match(team, num_players)
            if isinstance(r, str) or isinstance(r, list) or r is None:
                return r
            self._error = 'new_match needs to return kart names as a str, list, or None. Got {!r}!'.format(r)
        except Exception as e:
            self._error = 'Failed to start new_match: {}'.format(str(e))
        return []

    def act(self, player_state, *args, **kwargs):
        from time import time
        t0 = time()
        try:
            r = self._team.act(player_state, *args, **kwargs)
        except Exception as e:
            self._error = 'Failed to act: {}'.format(str(e))
        else:
            self._total_act_time += time()-t0
            return r
        return []

    def info(self):
        return RunnerInfo(self.agent_type, self._error, self._total_act_time)


class MatchException(Exception):
    def __init__(self, score, msg1, msg2):
        self.score, self.msg1, self.msg2 = score, msg1, msg2


class Match:
    """
        Do not create more than one match per process (use ray to create more)
    """
    def __init__(self, use_graphics=True, logging_level=None):
        # DO this here so things work out with ray
        import pystk
        self._pystk = pystk
        if logging_level is not None:
            logging.basicConfig(level=logging_level)

        # Fire up pystk
        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)

    def __del__(self):
        if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
            self._pystk.clean()

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    @classmethod
    def _r(cls, f):
        if hasattr(f, 'remote'):
            return f.remote
        if hasattr(f, '__call__'):
            if hasattr(f.__call__, 'remote'):
                return f.__call__.remote
        return f

    @staticmethod
    def _g(f):
        from .remote import ray
        if ray is not None and isinstance(f,  ray._raylet.ObjectRef):
            return ray.get(f)
        return f

    def _check(self, team1, team2, where, n_iter, timeout):
        _, error, t1 = self._g(self._r(team1.info)())
        if error:
            raise MatchException([0, 3], 'other team crashed', 'crash during {}: {}'.format(where, error))

        _, error, t2 = self._g(self._r(team2.info)())
        if error:
            raise MatchException([3, 0], 'crash during {}: {}'.format(where, error), 'other team crashed')

        logging.debug('timeout {} <? {} {}'.format(timeout, t1, t2))
        return t1 < timeout, t2 < timeout

    def run(self, team1, team2, num_player=1, max_frames=MAX_FRAMES, max_score=3, record_fn=None, timeout=1e10,
            initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], verbose=False):
        from .remote import ray
        from . import utils
        RaceConfig = self._pystk.RaceConfig

        logging.info('Creating teams')

        # Start a new match
        t1_cars = self._g(self._r(team1.new_match)(0, num_player)) or ['tux']
        t2_cars = self._g(self._r(team2.new_match)(1, num_player)) or ['tux']

        t1_type, *_ = self._g(self._r(team1.info)())
        t2_type, *_ = self._g(self._r(team2.info)())

        if t1_type == 'image' or t2_type == 'image':
            assert self._use_graphics, 'Need to use_graphics for image agents.'

        # Deal with crashes
        t1_can_act, t2_can_act = self._check(team1, team2, 'new_match', 0, timeout)

        # Setup the race config
        logging.info('Setting up race')

        race_config = RaceConfig(track=TRACK_NAME, mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * num_player)
        race_config.players.pop()
        for i in range(num_player):
            race_config.players.append(self._make_config(0, hasattr(team1, 'is_ai') and team1.is_ai, t1_cars[i % len(t1_cars)]))
            race_config.players.append(self._make_config(1, hasattr(team2, 'is_ai') and team2.is_ai, t2_cars[i % len(t2_cars)]))

        # Start the match
        logging.info('Starting race')
        race = self._pystk.Race(race_config)
        race.start()
        race.step()

        state = self._pystk.WorldState()
        state.update()
        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))

        for it in range(max_frames):
            logging.debug('iteration {} / {}'.format(it, MAX_FRAMES))
            state.update()

            # Get the state
            team1_state = [to_native(p) for p in state.players[0::2]]
            team2_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            team1_images = team2_images = None
            if self._use_graphics:
                team1_images = [np.array(race.render_data[i].image) for i in range(0, len(race.render_data), 2)]
                team2_images = [np.array(race.render_data[i].image) for i in range(1, len(race.render_data), 2)]

            # Have each team produce actions (in parallel)
            if t1_can_act:
                if t1_type == 'image':
                    team1_actions_delayed = self._r(team1.act)(team1_state, team1_images)
                else:
                    team1_actions_delayed = self._r(team1.act)(team1_state, team2_state, soccer_state)

            if t2_can_act:
                if t2_type == 'image':
                    team2_actions_delayed = self._r(team2.act)(team2_state, team2_images)
                else:
                    team2_actions_delayed = self._r(team2.act)(team2_state, team1_state, soccer_state)

            # Wait for the actions to finish
            team1_actions = self._g(team1_actions_delayed) if t1_can_act else None
            team2_actions = self._g(team2_actions_delayed) if t2_can_act else None

            new_t1_can_act, new_t2_can_act = self._check(team1, team2, 'act', it, timeout)
            if not new_t1_can_act and t1_can_act and verbose:
                print('Team 1 timed out')
            if not new_t2_can_act and t2_can_act and verbose:
                print('Team 2 timed out')

            t1_can_act, t2_can_act = new_t1_can_act, new_t2_can_act

            # Assemble the actions
            actions = []
            for i in range(num_player):
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
                actions.append(a1)
                actions.append(a2)

            if record_fn:
                self._r(record_fn)(team1_state, team2_state, soccer_state=soccer_state, actions=actions,
                                   team1_images=team1_images, team2_images=team2_images)
            logging.debug('  race.step  [score = {}]'.format(state.soccer.score))
            if (not race.step([self._pystk.Action(**a) for a in actions]) and num_player) or sum(state.soccer.score) >= max_score:
                break

        race.stop()
        del race
        #input(len(ray.get(record_fn.data.remote())))
        #print(ray.get(record_fn.data.remote()))
        if record_fn is utils.MultiRecorder():
            return record_fn._r[0].data()
        else:
            return ray.get(record_fn.data.remote())
        

    def wait(self, x):
        return x


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from os import environ
    from . import remote, utils
    from .remote import ray
    import pickle
    import random 
    parser = ArgumentParser(description="Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('-r', '--record_video', help="Do you want to record a video?")
    parser.add_argument('-s', '--record_state', help="Do you want to pickle the state?")
    parser.add_argument('-f', '--num_frames', default=1200, type=int, help="How many steps should we play for?")
    parser.add_argument('-p', '--num_players', default=2, type=int, help="Number of players per team")
    parser.add_argument('-m', '--max_score', default=5, type=int, help="How many goal should we play to?")
    parser.add_argument('-j', '--parallel', type=int, help="How many parallel process to use?")
    parser.add_argument('--ball_location', default=[0, 0], type=float, nargs=2, help="Initial xy location of ball")
    parser.add_argument('--ball_velocity', default=[0, 0], type=float, nargs=2, help="Initial xy velocity of ball")
    parser.add_argument('-n', '--filename', default="test.pkl", type=str, help="How many goal should we play to?")
    parser.add_argument('team1', help="Python module name or `AI` for AI players.")
    parser.add_argument('team2', help="Python module name or `AI` for AI players.")
    args = parser.parse_args()

    logging.basicConfig(level=environ.get('LOGLEVEL', 'WARNING').upper())

    if args.parallel is None or remote.ray is None:
        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else TeamRunner(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else TeamRunner(args.team2)
        # What should we record?
        recorder = None
        if args.record_video:
            recorder = recorder & utils.VideoRecorder(args.record_video)

        if args.record_state:
            recorder = recorder & utils.StateRecorder(args.record_state)
        recorder = recorder & utils.DataRecorder()
        # Start the match
        #train_data = []
        match = Match(use_graphics=True)
        for i in range(6):
            print(i)
            try:
                data = match.run(team1, team2, args.num_players, args.num_frames, max_score=args.max_score,
                                initial_ball_location=args.ball_location, initial_ball_velocity=args.ball_velocity,
                                record_fn=recorder)
                print(len(data))
            except MatchException as e:
                print('Match failed', e.score)
                print(' T1:', e.msg1)
                print(' T2:', e.msg2)
                train_data = torch.stack([torch.as_tensor(np.concatenate([d['team1_images'][0], d['team1_images'][1]], axis=2)) for d in ray.get(result)]).permute(0,3,1,2).to(device).float()/255.
                labels = torch.stack([torch.as_tensor([d['actions'][0]['acceleration'],d['actions'][0]['steer'],d['actions'][0]['brake'],d['actions'][2]['acceleration'],d['actions'][2]['steer'],d['actions'][2]['brake']]) for d in ray.get(result)]).to(device).float()
                print(train_images.shape)
                print(train_data.shape)
                train_images= torch.cat((train_images,train_data),dim=0)
                print(labels.shape)
                print(train_labels.shape)
                train_labels= torch.cat((train_labels,labels),axis=0)
       
                
    else:
        # Fire up ray
        remote.init(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()), configure_logging=True,
                    log_to_driver=True, include_dashboard=False)

        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else remote.RayTeamRunner.remote(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else remote.RayTeamRunner.remote(args.team2)
        team1_type, *_ = team1.info() if args.team1 == 'AI' else remote.get(team1.info.remote())
        team2_type, *_ = team2.info() if args.team2 == 'AI' else remote.get(team2.info.remote())

        # What should we record?
        assert args.record_state is None or args.record_video is None, "Cannot record both video and state in parallel mode"

        # Start the match
        for i in range(args.parallel):
            recorder = None
            if args.record_video:
                ext = Path(args.record_video).suffix
                recorder = remote.RayVideoRecorder.remote(args.record_video.replace(ext, f'.{i}{ext}'))
            elif args.record_state:
                ext = Path(args.record_state).suffix
                recorder = remote.RayStateRecorder.remote(args.record_state.replace(ext, f'.{i}{ext}'))
            recorder = remote.RayDataRecorder.remote()
            match = remote.RayMatch.remote(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()),
                                           use_graphics=True)
            print(i)
            #input(team1)
            result = match.run.remote(team1, team2, args.num_players, args.num_frames, max_score=args.max_score,
                                      initial_ball_location=args.ball_location,
                                      initial_ball_velocity=args.ball_velocity,
                                      record_fn=recorder)
            features=[]
            labels = []
            for d in ray.get(result):
                #print(d['team1_state'][0])
                data1= extract_featuresV2(d['team2_state'][0], d['soccer_state'], d['team1_state'], 0)
                data2= extract_featuresV2(d['team2_state'][1], d['soccer_state'], d['team1_state'], 0)
                label1 =torch.as_tensor([d['actions'][1]['acceleration'],d['actions'][1]['steer'],d['actions'][1]['brake']])
                label2 =torch.as_tensor([d['actions'][3]['acceleration'],d['actions'][3]['steer'],d['actions'][3]['brake']])

                #input(data1.shape)
                features.append(np.stack((data1, data2)))
                labels.append(np.stack((label1, label2)))
            train_data = torch.cat([torch.as_tensor(feature) for feature in features])
            grouped_labels = torch.cat([torch.as_tensor(label) for label in labels])
        
        
            
            if i == 0 :
                train_images = train_data.detach()
                train_labels = grouped_labels.detach()
            else:
                train_images= torch.cat((train_images,train_data),dim=0)
                train_labels= torch.cat((train_labels,grouped_labels),dim=0)
            print(train_images.shape)
            print(train_data.shape)
            print(grouped_labels .shape)
            print(train_labels.shape)

   
    #train_images = torch.tensor(train_images)
    #train_images = torch.stack([torch.as_tensor(np.concatenate([d['team1_images'][0], d['team1_images'][1]], axis=2)) for d in train_data]).permute(0,3,1,2).to(device).float()/255.
    #train_images = torch.stack([torch.as_tensor(np.concatenate([d['team1_images'][0], d['team1_images'][1]], axis=2)) for d in train_data]).permute(0,3,1,2).to(device).float()/255.
    #train_labels = torch.stack([torch.as_tensor([d['actions'][0]['acceleration'],d['actions'][0]['steer'],d['actions'][0]['brake'],d['actions'][2]['acceleration'],d['actions'][2]['steer'],d['actions'][2]['brake']]) for d in train_data]).to(device).float()
    #torch.save((train_images,train_labels),"imitation_even.pt",pickle_mode = 'ujson')
    with open(args.filename,'wb') as f:
        pickle.dump((train_images,train_labels),f)



