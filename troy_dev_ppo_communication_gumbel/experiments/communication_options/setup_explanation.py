"""Plain-language setup walkthrough, generated from the frozen implementations."""
from pathlib import Path
import torch
from .models import PrivateGoal,RouteSafety,Complementary


def setup_section(study,grid_figure):
    study=Path(study)
    def load(cls,method,task,seed):
        model=cls(method)
        model.load_state_dict(torch.load(study/f'{task}_{method}_{seed}'/'checkpoint.pt',
                                        weights_only=True,map_location='cpu')['model'])
        return model
    with torch.no_grad():
        goal=load(PrivateGoal,'learned','private_goal',0)
        z=int(goal.sender.argmax(-1)[0]);chosen=int(goal.receiver.argmax(-1)[z])
        goalword='silence' if z==0 else f'word {z}'
        route=load(RouteSafety,'learned','route_safety',0)
        states=[0,0,0,1,1,1,1,0,0];cache=0;trace=[]
        for t,state in enumerate(states):
            word=int(route.sender.argmax(-1)[t,state,cache])
            if word:cache=word
            choice=int(route.receiver.argmax(-1)[cache])
            trace.append(f"| {t} | {state} | {'silence' if word==0 else 'word '+str(word)} | {choice} | "
                         f"{'crossing succeeds' if t in (0,4,8) and choice==state else 'no crossing decision'} |")
        dialogue=load(Complementary,'dialogue_warm','complementary',2)
        q=3;world=[1,0,0,1,0,1,0,1]
        question=int(dialogue.query.argmax(-1)[q])
        slot=int(dialogue.lookup.argmax(-1)[question-1])
        reply=int(dialogue.bit_encoder.argmax(-1)[world[slot]])
        answer=int(dialogue.answer.argmax(-1)[q,reply])
    text='''<a id="simulated-setup"></a>

## Start here: the complete simulated setup

### What actually runs on the computer

An agent here is a learned decision function, not a rendered body with a camera.
The environment stores a small task state, such as a target index or a safe-side
bit. It exposes different parts of that state to two agents. They select public
symbols and task actions, and the environment scores whether the final decision
is correct. This is a deliberately small model of cooperative information use.

There are **two separate layers** in this study:

| Layer | What it contains | Role in the experiments |
|---|---|---|
| Communication decision process | Private facts, permitted observations, discrete messages, choices, memory and costs | The learned policies train and receive their primary scores here |
| Receiver-local navigation checks | Public grid, start cell, obstacles, destinations and a BFS path planner | A separate check shows how a selected destination or passage becomes movement |

**The communication learner does not move through the pictured grid during
training.** It learns the information-dependent choice that would tell a
receiver-local navigator where to go. The grid checks exercise that navigator
separately. There are no rendered lions, cages, food objects, camera images,
collisions integrated by a physics engine, or torque-controlled bodies in these
four experiments. Their target and safety variables stand in for the relevant
facts from the original MuJoCo setting. The earlier spatial PPO study is a
separate experiment and should not be conflated with this one.

@GRID@

### How to read the grid

The left map has 13 columns and 13 rows. The receiver starts at cell (6,6).
Four public destination locations are (1,1), (11,1), (1,11) and (11,11).
Dark cells are public obstacles. The example comes from the first saved
navigation-validation layout, including its actual blocked cells. The displayed
path goes to destination G2. The cells themselves do not reveal which destination
earns reward. That target identity is the sender's private fact.

The right map is one 7-by-7 room. A wall occupies x=3 except at y=1 and y=5.
The receiver starts at (0,3), and the exit is at (6,3). The two candidate passages
are numbered 0 and 1. Both locations are public. Their current safety is private
to the observer/helper. The two colored paths illustrate alternatives; the
receiver follows only the selected one. This is the public geometry of the
planner demonstration, not three physically simulated rooms in the learning loop.

The planner receives a start, a selected public waypoint, public obstacles and
map dimensions. It computes successive neighboring cells, changing x or y by
one each time. It has no access to the true rewarded target or safe-side label.
Navigation ability is supplied by this planner rather than learned. We therefore
isolate the information decision and do not claim to have trained locomotion.

### Who the agents are and what they can observe

The **observer/helper** owns environmental information. The **acting agent** must
complete the task. They are often called sender and receiver, respectively,
because the helper supplies the missing fact. In Options 3 and 4 the acting agent
can send a request first, so those names do not imply a fixed first speaker.

| Option | Helper's private input | Acting agent's permitted input | Acting agent's task choice |
|---|---|---|---|
| 1: Private goal | Which of four destinations is rewarded | Delivered opening word or a known silent slot | Commit to a destination index |
| 2: Changing safety | Current safe side, logical tick, and its predicted communication cache from its own history | Last delivered word in its cache | Choose passage 0 or 1 at a crossing |
| 3: Ask or scout | The true destination | Inspection cost, either the visible destination or an explicit unknown observation, and a reply if requested | Commit, inspect, or ask |
| 4: Complementary knowledge | Eight binary map facts and any received query word | Private entity index Q, its own query, and reply or no-reply observation | Choose the requested entity's safe entrance bit |

The target and safety labels never appear directly in the acting policy's input,
except in conditions explicitly granting observation or using an oracle. The
planner can separately use public coordinates after a waypoint is selected. The
decoder for Option 1, for example, sees a signal rather than the whole grid.
The helper in Option 4 receives a query code, not the acting agent's private Q.
In packet-loss tests the helper does not receive free delivery acknowledgments.

### What a message is

A public word is a discrete ID. Its numeric ID has no preassigned meaning.
The learned encoder and decoder establish a convention through shared reward.
Different seeds can assign different IDs to the same task fact.

Options 1 and 2 allow eight emitted words plus silence. A known silent time
slot can itself be informative. Silence in Option 2 preserves the last delivered
word. The cache is a supplied nine-state memory mechanism: empty or one of eight
words. These runs test use of that mechanism, not discovery of a memory architecture.

Option 3 has one fixed ASK request and one reply chosen from eight learned words.
The entire request/reply exchange costs 0.08. Option 4 has eight possible query
words and two possible answer words. A query word fits into three payload bits.
Labels 1 through 8 are just names for those eight alternatives. A presence bit
distinguishes no request, and the answer uses one further bit. An actual answer
bit with value zero is still a transmitted and charged bit, not silence.

### One episode of Option 1: selecting a private goal

1. The environment selects a destination uniformly from four possibilities.
   The helper observes its index. The acting agent does not.
2. The helper chooses an opening word or silence.
3. The acting agent decodes that signal into a destination index and commits.
4. A correct commitment earns 1. A wrong commitment earns 0 and ends the attempt.
   An emitted packet subtracts 0.05. There is no second trial-and-error guess.
5. In the separate path check, the receiver-local planner can turn the chosen
   index into a path to its public destination coordinates.

**Worked frozen-policy example:** with true target G0, learned seed 0 emits
@GOALWORD@ and the receiver selects G@GOALCHOICE@. This successful example has
utility 0.95 if a packet is emitted. The complete seed-0 code also uses silence
for one other target, which lowers average transmissions to 0.75 per episode.
These IDs come from the checkpoint, not a hand-written message dictionary.

There is no target-dependent intermediate reward or visual marker that lets a
silent agent discover the answer. Before communication, the four possible
target worlds are indistinguishable to it. This produces the 25% silent bound.

### One episode of Option 2: three safety decisions

One safe-side bit evolves across nine logical ticks, numbered 0 through 8.
It begins equally likely to be 0 or 1 and flips with probability 0.08 between
adjacent ticks. It is one correlated process, not three independent gate draws.
The acting agent must choose correctly at ticks 0, 4 and 8. A wrong choice
terminates the episode. Completing all three choices earns 1. Each attempted
packet before termination subtracts 0.02.

At a tick the helper first observes the current state and chooses whether to
send. Delivery then updates the receiver's cache. If this is a crossing tick,
the receiver chooses a side using that cache. If the episode continues, the
environment applies the possible state flip for the next tick. A logical tick
has no assigned duration in seconds and is not a MuJoCo frame or a grid move.

The following selected history executes the frozen learned seed-0 policy:

| Tick | True safe side | Helper signal | Receiver's decoded choice | Consequence |
|---|---:|---|---:|---|
@ROUTETABLE@

At a non-crossing tick, an outdated decoded choice does not yet incur failure.
The update must arrive before the next consequential decision. This is why the
agent can remain silent between crossings. Its passage choice could then direct
the local room planner shown above. The primary training score, however, comes
from the three logical decisions, not from executing their physical paths.

### One episode of Option 3: acting, inspecting, or asking

The environment independently selects a target, whether the acting agent can
already observe it, and an inspection cost. The target is uniform over four
possibilities. It is visible in half of episodes. Cost is uniform over 0.02,
0.12 and 0.40, giving 24 equally likely target/visibility/cost combinations.

The first action has six alternatives: commit to one of four targets, SCOUT,
or ASK. SCOUT purchases the correct target identity with certainty and charges
the specified inspection cost. **It is an abstract information-acquisition
action; no journey to a viewpoint is simulated and no travel energy is measured.**
ASK sends the fixed request and obtains a learned categorical reply. It costs
0.02 for the request and 0.06 for the reply. The reply decoder then selects a
target. There is one request/answer exchange, not an unrestricted conversation.

For a hidden target with inspection cost 0.40, a correct asked-for answer earns
utility 1−0.08=0.92; inspection earns 1−0.40=0.60. With cost 0.02, inspection
instead earns 0.98 and asking earns 0.92. When the target is already visible,
the receiver can act directly for utility 1. The desired policy therefore
changes with its own observation and the cost of acquiring information.

### One episode of Option 4: combining a private goal with a private map

The acting agent privately receives Q, one of eight entity indices. The helper
privately receives eight independent binary facts X. Each fact specifies the
safe entrance for one entity. The acting agent needs only XQ, while the helper
initially does not know which entity matters. The task averages all 256 maps
and all eight private indices, giving 2,048 equally likely contexts.

The acting agent can decline to ask or emit a query word. If it asks, the helper
uses that word to select a map slot, encodes its bit into an answer word, and
returns it. The acting agent combines the reply with its own Q to choose a bit.
The helper's lookup, the query code, and the answer interpretation are learned.
The existence of this lookup factorization is supplied by the architecture.

**Worked frozen-policy example:** seed 2 has Q=3 and map
X=[1,0,0,1,0,1,0,1]. Its query is word @QUESTION@. The helper's learned lookup
selects slot @SLOT@, whose fact is 1, and emits answer word @REPLY@. The acting
agent decodes entrance @ANSWER@. Answer word zero can therefore encode a fact
with value one under a learned convention. Word labels and task facts are
different quantities.

A requested exchange charges five bits: one presence flag, three query bits
and one answer bit. At 0.02 per bit, a correct requested answer gives utility
0.90. Declining still charges the protocol's one presence bit. The fully silent
reference disables the protocol entirely and has zero communication cost.
The full-map broadcast reference sends eight bits for perfect accuracy at
utility 0.84. The separately learned one-way compression reference has a fixed
five-bit message and never observes Q.

### What resets, what learns, and what the reported scores mean

An episode resets task facts and any communication cache. Learned weights
persist across episodes. Each optimization seed initializes a new set of
probability tables. There are no pretrained language models or visual models.

The tables are small categorical policies. Their entries are trainable logits,
which softmax turns into probabilities. Training updates the helper and actor
jointly to increase expected success after resource charges. The environment
provides a reward function, not a list of correct public words. An initially
arbitrary convention becomes useful when the receiver's interpretation and the
helper's production align.

**One update is not one sampled episode.** The optimizer sums probabilities of
possible states, symbols and actions and differentiates that expected reward.
The route task propagates surviving probability mass across its nine ticks.
This is exact known-model optimization, not sampled PPO. All logical task states
are available in training, so the results do not establish generalization to
unseen semantic states. Extra warm-up learning supplies a useful reply code
for warm scouting and a learned reference convention for warm dialogue.

Final scoring freezes the tables. The primary score uses each component's
highest-probability choice. A secondary score keeps sampling from its learned
probabilities. Exact population expectations are primary; one million separate
sampled executions check the evaluator. The 128 new map layouts exercise the
planner separately. They do not turn the finite communication study into a
joint spatial reinforcement-learning experiment.

### Why this stage is outside MuJoCo

The immediate question is whether the information structure makes communication
useful and whether the agents can learn when to use it. Separating that question
from visual recognition and movement execution gives us several controls:

- **Information access is explicit.** We can verify that a target or safe-side
  fact reaches only its owner, and derive the best silent success probability.
  A rendered scene can accidentally reveal the answer through a marker, object
  placement, or sensor field.
- **Failures have fewer possible causes.** If a receiver makes a wrong choice,
  we can distinguish a poor protocol from a missing fact. There is no additional
  uncertainty about image recognition, localization, locomotion, or collisions.
- **Counterfactuals are exact.** We can block, change, delay in an extended model,
  or lose packets and inspect the resulting decision process. The current
  implemented interventions are specified later; delay is not tested here.
- **Optimization and baselines are auditable.** The finite state spaces let us
  enumerate alternatives and calculate bounds. This would be much harder for
  continuous physical states with raw image observations.
- **Repeated runs are inexpensive.** We can check several starts and inspect
  complete code tables without rendering scenes or loading perception assets.
  This is a practicality argument, not a measured speed comparison with MuJoCo.

MuJoCo itself was not the reason the previous task lacked a communication
advantage. A cheap conservative route removed much of the need for information.
The same task-design problem can exist with or without physics. Conversely,
these redesigned information gaps could be recreated inside MuJoCo.

The simplification also removes difficulties we eventually need to test:
perception errors, observation latency, continuous geometry, uncertain motion,
physical feasibility, and time available to react. Success here supplies a
controlled demonstration under stated assumptions; it does not establish
successful or safe behavior in the original physical scene. Changing both the
task and the learner from the earlier PPO study also prevents attributing the
improvement to a single cause.

### How the result would map back to MuJoCo

1. Keep the redesigned private facts, permitted observations, available choices,
   and resource costs while replacing tables with a neural communication policy
   trained from sampled structured observations.
2. Recreate a task in MuJoCo with oracle structured observations. A receiver-local
   controller executes the selected destination or passage. Test task success
   and movement failures separately.
3. Replace oracle facts with learned perception. The helper must detect actual
   rendered state changes, while the actor retains only its permitted sensors.
4. Add and measure realistic detection errors, channel delay, action delay and
   motion uncertainty. Re-evaluate the protocol and the silent baselines.

Transfer requires more than matching the picture. The task-relevant facts,
observation asymmetry, time of important decisions, alternative ways to obtain
information, and reward/cost definitions must remain comparable. Those are
conditions to test, not an equivalence already established by this study.

'''
    replacements={'@GRID@':grid_figure,'@GOALWORD@':goalword,'@GOALCHOICE@':str(chosen),
                  '@ROUTETABLE@':'\n'.join(trace),'@QUESTION@':str(question),'@SLOT@':str(slot),
                  '@REPLY@':str(reply),'@ANSWER@':str(answer)}
    for key,value in replacements.items():text=text.replace(key,value)
    return text
