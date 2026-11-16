"""Receiver-local path planner. Its interface accepts no private task labels."""
from collections import deque


def plan(start,goal,blocked=(),width=13,height=13):
    blocked=set(blocked)
    frontier=deque([start]);parent={start:None}
    while frontier:
        cell=frontier.popleft()
        if cell==goal:
            path=[]
            while cell is not None:
                path.append(cell);cell=parent[cell]
            return path[::-1]
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt=(cell[0]+dx,cell[1]+dy)
            if 0<=nxt[0]<width and 0<=nxt[1]<height and nxt not in blocked and nxt not in parent:
                parent[nxt]=cell;frontier.append(nxt)
    raise ValueError('Unreachable public waypoint')


def room_route(choice):
    """Public two-gate room; private passability is checked by the task, not planner."""
    wall={(3,y) for y in range(7) if y not in (1,5)}
    waypoint=(3,1 if choice==0 else 5)
    return plan((0,3),waypoint,wall,7,7)+plan(waypoint,(6,3),wall,7,7)[1:]
