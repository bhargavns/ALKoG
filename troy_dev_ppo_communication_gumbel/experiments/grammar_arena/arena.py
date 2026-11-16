"""Extend the existing arena asset and receiver movement, without a grid executor.

The original CommunicationKGWorldEnv supplies the egocentric receiver motion
and MuJoCo position/forward update. This subclass adds four food candidates,
three public passages, a tiger and swept obstacle/hazard checks. It intentionally
uses exact public geometry. No renderer or camera observation enters a policy.
"""
from pathlib import Path
import copy,hashlib,math,re,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from environment.KGWorldEnv import CommunicationKGWorldEnv,ARENA_HALF,FOOD_REACH_RADIUS

# parents[2] is the project root for experiments/grammar_arena/arena.py.
BASE_ASSET=Path(__file__).resolve().parents[2]/'environment/assets/kg_world.xml'
FOOD_X=4.8
FOOD_Y=np.array([-4.2,-1.8,.6,3.])
GATE_Y=np.array([-4.2,-.6,4.2])
STEP=.20
BODY_RADIUS=.20
WALL_HALF=.16
GATE_HALF=.72
HAZARD_RADIUS=.78
WAYPOINT_REACH=.18
FOOD_BONUS=10.
HAZARD_PENALTY=50.
MOVE_COST=.01
BIT_COST=.001

def public_waypoints(food,route,include_start=False):
    """Public topological route with dense passage waypoints for clearance."""
    y=GATE_Y[route]
    points=[[-1.1,y],[-.45,y],[0.,y],[.45,y],[1.1,y],[3.4,y],
            [3.4,FOOD_Y[food]],[FOOD_X,FOOD_Y[food]]]
    # A rotated four-action frame can trace a wide L1 dog-leg toward a distant
    # waypoint. Public interpolation bounds that deviation near walls/food.
    # The controller still selects every .2 m movement; no pose is assigned
    # from these points and no action sequence is provided by a teacher.
    dense=[];previous=np.array([-4.8,-.6])
    for point in points:
        point=np.array(point);n=max(1,int(np.ceil(np.linalg.norm(point-previous)/.35)))
        dense.extend(previous+(point-previous)*i/n for i in range(1,n+1));previous=point
    return np.array(([[-4.8,-.6]] if include_start else [])+dense)

def _scale_body(body,factor):
    for e in body.iter():
        for attr in ('pos','size','fromto'):
            if attr in e.attrib:e.set(attr,' '.join(str(float(x)*factor) for x in e.get(attr).split()))

def asset_xml():
    # The legacy asset has human comments containing '--', accepted by MuJoCo
    # but invalid for ElementTree. Ignore comments; preserve every XML element.
    source=re.sub(r'<!--.*?-->','',BASE_ASSET.read_text(),flags=re.S)
    root=ET.fromstring(source);world=root.find('worldbody')
    root.set('model','kg_world_grammar_transfer')
    root.find('visual/global').set('offwidth','960');root.find('visual/global').set('offheight','720')
    ET.SubElement(world,'camera',name='grammar_top',pos='0 0 22',xyaxes='1 0 0 0 1 0',fovy='43')
    receiver=world.find("body[@name='receiver']");_scale_body(receiver,.45)
    lion=world.find("body[@name='lion']");_scale_body(lion,.7)
    tiger=copy.deepcopy(lion);tiger.set('name','tiger')
    for e in tiger.iter():
        if e is not tiger and 'name' in e.attrib:e.set('name','tiger_'+e.get('name'))
        if e.tag=='geom':e.set('rgba','.95 .65 .10 1')
    for x in (-.2,0,.2):ET.SubElement(tiger,'geom',name='stripe_'+str(x),type='capsule',fromto=f'{x} -.27 .25 {x} .27 .50',size='.025',rgba='.12 .10 .07 1',contype='0',conaffinity='0')
    world.append(tiger)
    cage=world.find("body[@name='cage']");_scale_body(cage,.38)
    cage2=copy.deepcopy(cage);cage2.set('name','tiger_cage')
    for e in cage2.iter():
        if e is not cage2 and 'name' in e.attrib:e.set('name','tiger_'+e.get('name'))
    world.append(cage2)
    old=world.find("body[@name='food']")
    for child in list(old):old.remove(child)
    for j in range(4):
        food=old if j==0 else ET.SubElement(world,'body',name=f'food_{j}')
        # All alternatives are present as collision-free visuals; only the
        # currently assigned kind's alpha is enabled on reset.
        for k in range(4):
            group=ET.SubElement(food,'body',name=f'food{j}_kind{k}')
            if k==0:ET.SubElement(group,'geom',name=f'f{j}k{k}',type='sphere',pos='0 0 .28',size='.28',contype='0',conaffinity='0')
            elif k==1:
                for q,(x,y) in enumerate(((-.15,0),(.15,0),(0,.17))):ET.SubElement(group,'geom',name=f'f{j}k{k}_{q}',type='sphere',pos=f'{x} {y} .22',size='.18',contype='0',conaffinity='0')
            elif k==2:ET.SubElement(group,'geom',name=f'f{j}k{k}',type='ellipsoid',pos='0 0 .18',size='.38 .16 .16',contype='0',conaffinity='0')
            else:
                ET.SubElement(group,'geom',name=f'f{j}k{k}_stem',type='cylinder',pos='0 0 .18',size='.10 .18',contype='0',conaffinity='0')
                ET.SubElement(group,'geom',name=f'f{j}k{k}_cap',type='ellipsoid',pos='0 0 .40',size='.35 .35 .12',contype='0',conaffinity='0')
    edges=[(-10.,GATE_Y[0]-GATE_HALF),(GATE_Y[0]+GATE_HALF,GATE_Y[1]-GATE_HALF),
           (GATE_Y[1]+GATE_HALF,GATE_Y[2]-GATE_HALF),(GATE_Y[2]+GATE_HALF,10.)]
    for j,(lo,hi) in enumerate(edges):
        body=ET.SubElement(world,'body',name=f'passage_wall{j}',pos=f'0 {(lo+hi)/2} .7')
        ET.SubElement(body,'geom',name=f'wall{j}',type='box',size=f'{WALL_HALF} {(hi-lo)/2} .7',rgba='.32 .36 .4 1',contype='0',conaffinity='0')
    return ET.tostring(root,encoding='unicode'),edges

XML,WALL_INTERVALS=asset_xml()

def segment_aabb(p,q,lo,hi):
    """Inclusive swept segment intersection, including a zero-length segment."""
    p=np.asarray(p);q=np.asarray(q);lo=np.asarray(lo);hi=np.asarray(hi);v=q-p;t0,t1=0.,1.
    for a in range(2):
        if abs(v[a])<1e-12:
            if p[a]<lo[a] or p[a]>hi[a]:return False
        else:
            t=(lo[a]-p[a])/v[a];u=(hi[a]-p[a])/v[a]
            t0=max(t0,min(t,u));t1=min(t1,max(t,u))
            if t0>t1:return False
    return True

def segment_distance(p,q,point):
    v=q-p;den=float(v@v);t=np.clip(float((point-p)@v)/den,0,1) if den>1e-16 else 0.
    return float(np.linalg.norm(p+t*v-point))

class FoodArena(CommunicationKGWorldEnv):
    """Same arena and inherited movement; task reset/scoring are specialized."""
    def __init__(self,seed=0):
        # Avoid creating dozens of OpenGL contexts for nonvisual training.
        # The original asset is extended before compilation; no base file edits.
        self.model=mujoco.MjModel.from_xml_string(XML);self.data=mujoco.MjData(self.model)
        self.renderer=None;self.rng=np.random.default_rng(seed)
        self.step_size=STEP;self.colocated=False;self.receiver_yaw=0.;self.render_size=720
        self.scale=1.;self.rotation=np.eye(2);self.translation=np.zeros(2);self.walls_enabled=True
        self.food_bodies=[self.model.body('food' if j==0 else f'food_{j}').id for j in range(4)]
        self.receiver_id=self.model.body('receiver').id
        self.wall_ids=[self.model.body(f'passage_wall{j}').id for j in range(4)]
        self.kind_geom_ids=[[np.array([g for g in range(self.model.ngeom) if self.model.geom_bodyid[g]==self.model.body(f'food{j}_kind{k}').id]) for k in range(4)] for j in range(4)]
        self.bump=False;self.steps=0;self.facts=np.zeros(4,dtype=np.int64);self.swap=0
        self.candidates=np.zeros((4,2),dtype=np.int64);self.target=0;self.mode='task';self.horizon=180
        self.waypoints=[];self.phase=0;self.selected_food=0;self.selected_route=0
        self.reset_motor()

    def world(self,points):return np.asarray(points)@self.rotation.T*self.scale+self.translation
    def local(self,points):return (np.asarray(points)-self.translation)@self.rotation/self.scale

    def _geometry(self,scale=1.,translation=(0,0),rotation=0.,walls=True):
        self.scale=float(scale);self.translation=np.array(translation,float)
        theta=float(rotation);self.rotation=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        self.walls_enabled=walls
        quat=[math.cos(theta/2),0,0,math.sin(theta/2)]
        for body,(lo,hi) in zip(self.wall_ids,WALL_INTERVALS):
            self.model.body_pos[body]=[*self.world([0,(lo+hi)/2]),.7 if walls else -100]
            self.model.body_quat[body]=quat
            geom=np.flatnonzero(self.model.geom_bodyid==body)[0]
            self.model.geom_size[geom,:2]=[WALL_HALF*self.scale,(hi-lo)*self.scale/2]

    def _place(self,name,xy,z=0):self.model.body(name).pos[:]=[*xy,z]

    def _set_receiver(self,xy,yaw):
        self._place('receiver',xy);self.receiver_yaw=float(yaw)
        self.data.qpos[:]=0.;self.data.qvel[:]=0.;self.data.qpos[:2]=[-6.3,6.3]
        for name in ('water','poison'):self._place(name,[-20,-20],-100)
        mujoco.mj_forward(self.model,self.data)

    def reset_motor(self):
        self.mode='motor';self.steps=0;self.horizon=90;self._geometry(walls=False)
        # All distances/directions, including final corrections, are public.
        while True:
            start=self.rng.uniform(-5.3,5.3,2);goal=self.rng.uniform(-5.3,5.3,2)
            if .32<np.linalg.norm(goal-start)<9.:break
        self.motor_goal=goal;self._set_receiver(start,self.rng.uniform(-math.pi,math.pi))
        for name in ('lion','tiger','cage','tiger_cage'):self._place(name,[-20,-20],-100)
        for body in self.food_bodies:self.model.body_pos[body]=[20,20,-100]
        mujoco.mj_forward(self.model,self.data)
        return self.observation()

    def reset_task(self,facts,candidates,swap,geometry,yaw):
        self.mode='task';self.steps=0;self.horizon=180
        self.facts=np.array(facts,dtype=np.int64);self.candidates=np.array(candidates,dtype=np.int64);self.swap=int(swap)
        self.target=int(np.flatnonzero((self.candidates==self.facts[:2]).all(-1))[0])
        self._geometry(**geometry);self._set_receiver(self.world([-4.8,-.6]),yaw)
        self.food_xy=self.world(np.stack((np.full(4,FOOD_X),FOOD_Y),-1));self.gate_xy=self.world(np.stack((np.zeros(3),GATE_Y),-1))
        for body,xy in zip(self.food_bodies,self.food_xy):self.model.body_pos[body]=[*xy,0]
        colours=((.85,.15,.12),(.12,.75,.22),(.95,.85,.12),(.15,.35,.9))
        for j,(kind,colour) in enumerate(self.candidates):
            for k,ids in enumerate(self.kind_geom_ids[j]):self.model.geom_rgba[ids]=[*colours[colour],float(k==kind)]
        lion_gate=0 if self.swap==0 else 1;tiger_gate=1-lion_gate
        self._place('lion',self.gate_xy[lion_gate]);self._place('tiger',self.gate_xy[tiger_gate])
        for species,name,g in ((0,'cage',lion_gate),(1,'tiger_cage',tiger_gate)):
            free=self.facts[2]==species and self.facts[3]==1
            self._place(name,self.gate_xy[g],-100 if free else 0)
        self.hazard_xy=None if not self.facts[3] else self.gate_xy[0 if self.swap==self.facts[2] else 1].copy()
        self.phase=0;self.waypoints=[];self.bump=False
        mujoco.mj_forward(self.model,self.data)

    def select(self,food,route):
        """Public passage waypoints, not BFS or a learned planning claim."""
        self.selected_food=int(food);self.selected_route=int(route)
        local=public_waypoints(food,route)
        self.waypoints=self.world(local);self.phase=0
        return self.observation()

    def observation(self):
        goal=self.motor_goal if self.mode=='motor' else self.waypoints[self.phase]
        delta=goal-self.receiver_xy
        c,s=math.cos(self.receiver_yaw),math.sin(self.receiver_yaw)
        # Egocentric waypoint offset only. Private facts and the target index
        # cannot enter this motor policy; even absolute own position is absent.
        local_delta=np.array([[c,s],[-s,c]])@delta
        distance=float(np.linalg.norm(local_delta))
        return np.array([*(local_delta/max(distance,1e-8)),distance/8],dtype=np.float32)

    def _blocked(self,p,q):
        if not self.walls_enabled:return False
        p,q=self.local(p),self.local(q);r=BODY_RADIUS/self.scale
        return any(segment_aabb(p,q,[-WALL_HALF-r,lo-r],[WALL_HALF+r,hi+r]) for lo,hi in WALL_INTERVALS)

    def _apply_action(self,action):
        old=self.receiver_xy
        # Actual inherited existing-arena displacement plus mj_forward.
        super()._apply_action(action)
        new=self.receiver_xy;self.bump=self._blocked(old,new)
        if self.bump:self._place('receiver',old);mujoco.mj_forward(self.model,self.data)

    def step(self,action):
        old=self.receiver_xy;goal=self.motor_goal if self.mode=='motor' else self.waypoints[self.phase]
        before=float(np.linalg.norm(old-goal));self._apply_action(action);self.steps+=1
        new=self.receiver_xy;after=float(np.linalg.norm(new-goal))
        if self.mode=='motor':
            reached=after<WAYPOINT_REACH
            terminated=bool(reached);truncated=self.steps>=self.horizon and not terminated
            reward=.8*(before-after)-MOVE_COST+2.*reached-.05*self.bump
            return self.observation(),float(reward),terminated,truncated,dict(reached=bool(reached),hazard=False,bump=self.bump,food=-1,phase=0)
        caught=self.hazard_xy is not None and segment_distance(old,new,self.hazard_xy)<HAZARD_RADIUS
        # A physical encounter with ANY food terminates. The public approach
        # corridor keeps paths away from distractor food, rather than disabling
        # wrong-food detection or teleporting to the chosen target.
        touched=[j for j,xy in enumerate(self.food_xy) if segment_distance(old,new,xy)<FOOD_REACH_RADIUS]
        food=touched[0] if touched and not caught else -1
        reached=food==self.target and food>=0
        terminated=bool(caught or food>=0);truncated=self.steps>=self.horizon and not terminated
        reward=-MOVE_COST-HAZARD_PENALTY*caught+FOOD_BONUS*reached
        if not terminated and not truncated:
            while self.phase<len(self.waypoints)-1 and np.linalg.norm(new-self.waypoints[self.phase])<WAYPOINT_REACH:self.phase+=1
        return self.observation(),float(reward),terminated,truncated,dict(reached=bool(reached),hazard=bool(caught),bump=self.bump,food=food,phase=self.phase)

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None

    def render_top(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=720,width=960)
        self.renderer.update_scene(self.data,camera='grammar_top');return self.renderer.render().copy()

def source_identity():
    from environment import KGWorldEnv
    return dict(base_asset_sha256=hashlib.sha256(BASE_ASSET.read_bytes()).hexdigest(),
                base_environment_sha256=hashlib.sha256(Path(KGWorldEnv.__file__).read_bytes()).hexdigest(),
                movement='inherited CommunicationKGWorldEnv._apply_action; added swept wall validation',
                dynamics='kinematic position updates and mujoco.mj_forward; no torque locomotion')
