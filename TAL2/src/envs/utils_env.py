import os
import dgl
import math
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

current_milli_time = lambda: int(round(time.time() * 1000))


def euler_to_quaternion(rpy):
    """Convert [roll, pitch, yaw] to quaternion [x, y, z, w]."""
    if rpy is None or len(rpy) != 3:
        raise ValueError("Euler angles must be a 3-element iterable [roll, pitch, yaw].")
    roll, pitch, yaw = [float(v) for v in rpy]
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def orientation_to_quaternion(orientation):
    if orientation is None:
        return [0.0, 0.0, 0.0, 1.0]
    if len(orientation) == 4:
        return [float(v) for v in orientation]
    if len(orientation) == 3:
        return euler_to_quaternion(orientation)
    return [0.0, 0.0, 0.0, 1.0]


def _extract_pose(reference):
    """
    Normalize a pose reference into ([x, y, z], [x, y, z, w]).
    This file no longer supports raw PyBullet body ids.
    """
    if isinstance(reference, dict):
        if "position" in reference:
            position = reference["position"]
            orientation = reference.get("orientation", [0.0, 0.0, 0.0, 1.0])
            return list(position), orientation_to_quaternion(orientation)
    if isinstance(reference, (list, tuple)) and len(reference) == 2:
        position, orientation = reference
        if isinstance(position, (list, tuple)) and len(position) == 3:
            return list(position), orientation_to_quaternion(orientation)
    raise RuntimeError(
        "PyBullet body ids are no longer supported in utils_env.py. "
        "Pass explicit pose data instead."
    )


def _raise_legacy_backend_removed(function_name):
    raise RuntimeError(
        f"{function_name} depended on PyBullet runtime APIs and has been disabled during "
        "the Isaac Lab migration."
    )


def initDisplay(display):
    plt.axis('off')
    plt.rcParams['figure.figsize'] = [8, 6]
    cam = plt.figure()
    plt.axis('off')
    ax = plt.gca()
    return ax, cam


def initLogging():
    plt.axis('off')
    fig = plt.figure(figsize=(38.42, 21.6))
    return fig


names = {}


def keepHorizontal(object_list):
    """Keep the objects horizontal."""
    _raise_legacy_backend_removed("keepHorizontal")


def keepOnGround(object_list):
    """Keep the objects on ground."""
    _raise_legacy_backend_removed("keepOnGround")


def keepOrientation(objects):
    """keeps the orientation fixed."""
    _raise_legacy_backend_removed("keepOrientation")


def moveKeyboard(x1, y1, o1, object_list):
    """Move robot based on keyboard inputs."""
    del object_list
    _raise_legacy_backend_removed("moveKeyboard")


def moveUR5Keyboard(robotID, wings, gotoWing):
    """Change UR5 arm position based on keyboard input."""
    del robotID, wings, gotoWing
    _raise_legacy_backend_removed("moveUR5Keyboard")


def changeCameraOnKeyboard(camDistance, yaw, pitch, x, y):
    """Change camera zoom or angle from keyboard."""
    del x, y
    _raise_legacy_backend_removed("changeCameraOnKeyboard")


def changeCameraOnInput(camDistance, yaw, deltaDistance, deltaYaw):
    """Change camera zoom or angle from input."""
    return (camDistance + 0.5 * deltaDistance, yaw + 5 * deltaYaw)


def mentionNames(id_lookup):
    """Add labels of all objects in the world."""
    del id_lookup
    _raise_legacy_backend_removed("mentionNames")


def getAllPositionsAndOrientations(id_lookup):
    """Get position and orientation of all objects for data."""
    metrics = dict()
    for obj in id_lookup.keys():
        metrics[obj] = _extract_pose(id_lookup[obj])
    return metrics


def restoreOnKeyboard(world_states, x1, y1, o1):
    """Restore to last saved state when 'r' is pressed."""
    del world_states, x1, y1, o1
    _raise_legacy_backend_removed("restoreOnKeyboard")


def restoreOnInput(world_states, x1, y1, o1, constraints):
    """
    Restore to last saved state when this function is called.
    # * Modified by Xianqi Zhang. Data: 2021.11.02
    """
    del world_states, x1, y1, o1, constraints
    _raise_legacy_backend_removed("restoreOnInput")


def isInState(enclosure, state, position):
    """Check if enclosure is closed or not."""
    positionAndOrientation = state
    q = orientation_to_quaternion(positionAndOrientation[1])
    if len(position[1]) == 4:
        ((x1, y1, z1), (a1, b1, c1, d1)) = position
    elif len(position[1]) == 3:
        ((x1, y1, z1), (a1, b1, c1, d1)) = [position[0], orientation_to_quaternion(position[1])]
    else:
        ((x1, y1, z1), (a1, b1, c1, d1)) = [position[0], q]
    ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
    closed = (abs(x2 - x1) <= 0.07 and
              abs(y2 - y1) <= 0.07 and
              abs(z2 - z1) <= 0.07 and
              abs(a2 - a1) <= 0.07 and
              abs(b2 - b2) <= 0.07 and
              abs(c2 - c1) <= 0.07 and
              abs(d2 - d2) <= 0.07)
    return closed


def findConstraintTo(obj1, constraints):
    if obj1 in constraints.keys() and len(constraints[obj1]) != 0:
        return constraints[obj1][0]
    for holder, attached in constraints.items():
        if isinstance(attached, (list, tuple)) and obj1 in attached:
            return holder
    return ''


def findConstraintWith(obj1, constraints):
    l = []
    for obj in constraints.keys():
        if obj1 in constraints[obj][0]:
            l.append(obj)
    return l


# * Source function.
# def checkGoal(goal_file, constraints, states, id_lookup, on, clean, sticky, fixed, drilled, welded, painted):
#     """Check if goal conditions are true for the current state."""
#     if not goal_file:
#         return False
#     with open(goal_file, 'r') as handle:
#         file = json.load(handle)
#     goals = file['goals']
#     success = True
#
#     for goal in goals:
#         obj = goal['object']
#         if obj == 'light':
#             if obj in on:
#                 success = False
#
#         if obj == 'generator':
#             if not obj in on:
#                 success = False
#
#         if 'part' in obj:
#             success = success and obj in welded and obj in painted
#
#         if 'paper' in obj and goal['state'] == '':
#             tgt = findConstraintWith(obj, constraints)
#             heavy = False
#             for t in tgt:
#                 if not (t == '' or 'paper' in t):
#                     heavy = True
#             success = success and heavy
#
#         if obj == 'dirt' or obj == 'water' or obj == 'oil':
#             success = success and obj in clean
#
#         if goal['target'] != '':
#             tgt = findConstraintTo(obj, constraints)
#             while not (tgt == '' or tgt == goal['target']):
#                 tgt = findConstraintTo(tgt, constraints)
#             success = success and (tgt == goal['target'])
#
#         if goal['state'] != '':
#             finalstate = goal['state']
#             if finalstate == 'stuck' and not obj in sticky:
#                 success = False
#             if finalstate == 'fixed':
#                 finalstate = 'stuck'
#                 success = (success and (
#                         ('nail' in findConstraintWith(obj, constraints)
#                          and 'nail' in fixed) or
#                         ('screw' in findConstraintWith(obj, constraints)
#                          and 'screw' in fixed)))
#             st = states[obj][finalstate]
#             done = isInState(obj, st, p.getBasePositionAndOrientation(id_lookup[obj]))
#             success = success and done
#
#         if goal['position'] != '':
#             pos = p.getBasePositionAndOrientation(id_lookup[obj])[0]
#             goal_pos = p.getBasePositionAndOrientation(id_lookup[goal['position']])[0]
#             if abs(distance.euclidean(pos, goal_pos)) > abs(goal['tolerance']):
#                 success = False
#     return success


def checkUR5constrained(constraints):
    """Check if UR5 gripper is already holding something."""
    for obj in constraints.keys():
        if constraints[obj][0] == 'ur5':
            return True
    return False


def checkInside(constraints, states, id_lookup, obj, enclosures):
    """Check if object is inside cupboard or fridge."""
    for enclosure in enclosures:
        if not enclosure in id_lookup.keys(): continue
        if isClosed(enclosure, states, id_lookup):
            (x1, y1, z1), _ = _extract_pose(id_lookup[obj])
            (x2, y2, z2), _ = _extract_pose(id_lookup[enclosure])
            (l, w, h) = 1.0027969752543706, 0.5047863562602029, 1.5023976731489332
            inside = abs(x2 - x1) < 0.5 * l and abs(y2 - y1) < 1.5 * w and abs(z1 - z2) < 0.6 * h
            tgt = findConstraintTo(obj, constraints)
            # while not (tgt == '' or tgt == enclosure):
            if not (tgt == '' or tgt == enclosure):
                tgt = findConstraintTo(tgt, constraints)
            if inside or (tgt == enclosure): return True
    return False


def isClosed(enclosure, states, id_lookup):
    """Check if enclosure is closed or not."""
    positionAndOrientation = states[enclosure]['close']
    q = orientation_to_quaternion(positionAndOrientation[1])
    ((x1, y1, z1), (a1, b1, c1, d1)) = _extract_pose(id_lookup[enclosure])
    ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
    closed = (abs(x2 - x1) <= 0.01 and
              abs(y2 - y1) <= 0.01 and
              abs(a2 - a1) <= 0.01 and
              abs(b2 - b2) <= 0.01 and
              abs(c2 - c1) <= 0.01 and
              abs(d2 - d2) <= 0.01)
    return closed


def objDistance(obj1, obj2, id_lookup):
    (x, y, z), _ = _extract_pose(id_lookup[obj1])
    (x2, y2, z2), _ = _extract_pose(id_lookup[obj2])
    return math.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)


def saveImage(
        lastTime,
        imageCount,
        display,
        ax,
        o1,
        cam,
        dist,
        yaw,
        pitch,
        wall_id,
        on,
        camTargetPos=None,
        roll=-30,
        upAxisIndex=2,
        pixelWidth=1600,
        pixelHeight=1200,
        nearPlane=0.01,
        farPlane=100,
        fov=60
):
    del (lastTime, imageCount, display, ax, o1, cam, dist, yaw, pitch, wall_id, on,
         camTargetPos, roll, upAxisIndex, pixelWidth, pixelHeight, nearPlane, farPlane, fov)
    _raise_legacy_backend_removed("saveImage")


def deleteAll(path):
    filesToRemove = [os.path.join(path, f) for f in os.listdir(path)]
    for f in filesToRemove:
        os.remove(f)


def globalIDLookup(objs, objects):
    gidlookup = {}
    for i in range(len(objects)):
        if objects[i]['name'] in objs:
            gidlookup[objects[i]['name']] = i
    return gidlookup


def checkNear(obj1, obj2, metrics):
    (x1, y1, z1) = metrics[obj1][0]
    (x2, y2, z2) = metrics[obj2][0]
    return abs(distance.euclidean((x1, y1, z1), (x2, y2, z2))) < 3


def checkIn(obj1, obj2, obj1G, obj2G, metrics, constraints):
    if 'Container' in obj2G['properties']:
        if obj1 in ['cupboard', 'fridge']: return False
        (x1, y1, z1) = metrics[obj1][0]
        (x2, y2, z2) = metrics[obj2][0]
        (l, w, h) = obj2G['size']
        inside = abs(x2 - x1) < l and abs(y2 - y1) < 1.5 * w and abs(z1 - z2) < h
        tgt = findConstraintTo(obj1, constraints)
        if not (tgt == '' or tgt == obj2):
            tgt = findConstraintTo(tgt, constraints)
        return inside or (tgt == obj2)
    return False


def checkOn(obj1, obj2, obj1G, obj2G, metrics, constraints):
    if 'Surface' in obj2G['properties']:
        (x1, y1, z1) = metrics[obj1][0]
        (x2, y2, z2) = metrics[obj2][0]
        (l, w, h) = obj2G['size']
        on = abs(x2 - x1) < l + 0.2 and abs(y2 - y1) < w + 0.2 and z1 > z2 + 0.75 * h
        tgt = findConstraintTo(obj1, constraints)
        if not (tgt == '' or tgt == obj2):
            tgt = findConstraintTo(tgt, constraints)
        return on or (tgt == obj2)
    return False


def getDirectedDist(obj1, obj2, metrics):
    """Returns delX, delY, delZ, delO from obj1 to obj2."""
    (x1, y1, z1) = metrics[obj1][0]
    (x2, y2, z2) = metrics[obj2][0]
    return [x2 - x1, y2 - y1, z2 - z1, math.atan2((y2 - y1), (x2 - x1)) % (2 * math.pi)]


def grabbedObj(obj, constraints):
    """Check if object is grabbed by robot."""
    if obj in constraints.keys() and len(constraints[obj]) != 0 and constraints[obj][0] == 'ur5':
        return True
    held = constraints.get('husky', [])
    return isinstance(held, (list, tuple)) and obj in held


def getGoalObjects(world_name, goal_name):
    """Return set of objects in goal."""
    if 'home' in world_name:
        if goal_name == 'goal1-milk-fridge':
            return ['milk', 'fridge']
        elif goal_name == 'goal2-fruits-cupboard':
            return ['cupboard', 'apple', 'banana', 'orange']
        elif goal_name == 'goal3-clean-dirt':
            return ['dirt']
        elif goal_name == 'goal4-stick-paper':
            return ['paper', 'wall']
        elif goal_name == 'goal5-cubes-box':
            return ['box', 'cube_red', 'cube_green', 'cube_gray']
        elif goal_name == 'goal6-bottles-dumpster':
            return ['dumpster', 'bottle_blue', 'bottle_gray', 'bottle_red']
        elif goal_name == 'goal7-weight-paper':
            return ['paper']
        elif goal_name == 'goal8-light-off':
            return ['light']
    if 'factory' in world_name:
        if goal_name == 'goal1-crates-platform':
            return ['crate_green', 'crate_red', 'crate_peach', 'platform']
        elif goal_name == 'goal2-paper-wall':
            return ['paper', 'wall_warehouse']
        elif goal_name == 'goal3-board-wall':
            return ['board', 'wall_warehouse']
        elif goal_name == 'goal4-generator-on':
            return ['generator']
        elif goal_name == 'goal5-assemble-parts':
            return ['assembly_station', 'part1', 'part2', 'part3']
        elif goal_name == 'goal6-tools-workbench':
            return ['workbench', 'screwdriver', 'welder', 'drill']
        elif goal_name == 'goal7-clean-water':
            return ['water']
        elif goal_name == 'goal8-clean-oil':
            return ['oil']


def getPossiblePredicates(config, action):
    assert action in config.possibleActions
    if action == 'moveTo':
        return [config.property2Objects['all']]
    elif action == 'drop':
        return [config.property2Objects['Movable']]
    elif action == 'pick':
        return [config.property2Objects['Movable']]
    elif action == 'pushTo':
        return [config.property2Objects['Movable'], config.place_targets]
    elif action == 'pickNplaceAonB':
        return [config.property2Objects['Movable'], config.place_targets]


###################################################################################
# * Change to adapt the new goal json file format. Similar to function cg in approx.py.
def checkGoal(goal_file, constraints, states, id_lookup, on, clean, sticky, fixed, drilled, welded,
              painted):
    """Check if goal conditions are true for the current state."""
    if goal_file is None:
        # print('goal file is None')
        return False
    # with open(goal_file, 'r') as handle:
    #     file = json.load(handle)
    goals = goal_file['goals']
    success = True

    for goal in goals:
        obj = goal['object']

        # * Check state.
        if obj == 'light' or obj == 'generator':
            for tmp_state in goal['state']:
                if (tmp_state == 'off' and obj in on) or (tmp_state == 'on' and obj not in on):
                    return False

        elif 'part' in obj:
            if ('welded' in goal['state'] and not obj in welded) or (
                    'painted' in goal['state'] and not obj in painted):
                return False

        elif 'paper' in obj and len(goal['state']) == 0 and goal['position'] == '' and goal[
            'target'] == '':
            # * weight paper task: state=[], position='', target=''
            if obj not in constraints:
                return False
            tgt = findConstraintWith(obj, constraints)
            heavy = False
            for t in tgt:
                if not (t == '' or 'paper' in t):
                    heavy = True
            success = success and heavy

        elif obj == 'dirt' or obj == 'water' or obj == 'oil':
            if len(goal['state']) == 0:
                success = success and obj in clean
            else:
                for tmp_state in goal['state']:
                    if (tmp_state == 'clean' and (obj not in clean)) or (
                            tmp_state == 'dirty' and (obj in clean)):
                        return False

        elif len(goal['state']) != 0:
            for tmp_state in goal['state']:
                finalstate = tmp_state
                if finalstate == 'stuck' and obj not in sticky:  # * stick paper.
                    return False
                if finalstate == 'fixed':
                    finalstate = 'stuck'
                    success = (success and (
                            ('nail' in findConstraintWith(obj, constraints)
                             and 'nail' in fixed) or
                            ('screw' in findConstraintWith(obj, constraints)
                             and 'screw' in fixed)))
                st = states[obj][finalstate]
                done = isInState(obj, st, _extract_pose(id_lookup[obj]))
                success = success and done

        # * Check target.
        if goal['target'] != '':
            tgt = findConstraintTo(obj, constraints)
            # while not (tgt == '' or tgt == goal['target']):
            if not (tgt == '' or tgt == goal['target']):
                tgt = findConstraintTo(tgt, constraints)
            success = success and (tgt == goal['target'])

        # * Check position.
        if goal['position'] != '':
            pos = _extract_pose(id_lookup[obj])[0]
            goal_pos = _extract_pose(id_lookup[goal['position']])[0]
            if abs(distance.euclidean(pos, goal_pos)) > abs(goal['tolerance']):
                success = False

    return success


def convertToDGLGraph(config, graph_data, globalNode, globalID, ignore: list = None):
    """ Converts the graph from the datapoint into a DGL form of graph."""
    if ignore is None:
        ignore = []
    # * Make edge sets.
    close, inside, on, stuck = [], [], [], []
    closeToAgent = []
    for edge in graph_data['edges']:
        if edge['relation'] == 'Close':
            close.append((edge['from'], edge['to']))
            if edge['from'] == config.all_objects.index('husky'):
                closeToAgent.append(edge['to'])
        elif edge['relation'] == 'Inside':
            inside.append((edge['from'], edge['to']))
        elif edge['relation'] == 'On':
            on.append((edge['from'], edge['to']))
        elif edge['relation'] == 'Stuck':
            stuck.append((edge['from'], edge['to']))
    edgeDict = {
        ('object', 'Close', 'object'): close + [(i, i) for i in ignore],
        ('object', 'Inside', 'object'): inside,
        ('object', 'On', 'object'): on,
        ('object', 'Stuck', 'object'): stuck
    }
    if globalNode:
        globalList = []
        for i in range(globalID):
            globalList.append((i, globalID))
        edgeDict[('object', 'Global', 'object')] = globalList
    g = dgl.heterograph(edgeDict, num_nodes_dict={'object': len(config.all_objects)})
    # g = dgl.heterograph(edgeDict)
    # * Add node features
    n_nodes = len(config.all_objects)
    node_states = torch.zeros([n_nodes, config.N_STATES], dtype=torch.float)  # * State vector.
    node_vectors = torch.zeros([n_nodes, config.PRETRAINED_VECTOR_SIZE],
                               dtype=torch.float)  # * Fasttext embedding.
    node_size_and_pos = torch.zeros([n_nodes, 10], dtype=torch.float)  # * Size and position.
    node_close_agent = torch.zeros([n_nodes, 1], dtype=torch.float)  # * Close to husky.

    for i, node in enumerate(graph_data['nodes']):
        states = node['states']
        node_id = node['id']

        # * Filter high values to ignore the environment simulation error.
        node['position'] = list(node['position'])
        node['position'][0] = list(node['position'][0])
        if abs(node['position'][0][0]) >= 8.0:
            node['position'][0][0] = 0
        if abs(node['position'][0][1]) >= 8.0:
            node['position'][0][1] = 0
        if abs(node['position'][0][2]) >= 3.0:
            node['position'][0][2] = 0

        if node_id in ignore: continue
        for state in states:
            idx = config.state2indx[state]
            node_states[node_id, idx] = 1
        # 2026-05-18 修改：node['vector'] 在部分路径下已经是 tensor，
        # 这里改用 as_tensor 避免 torch.tensor(tensor) 反复构造触发 warning。
        node_vectors[node_id] = torch.as_tensor(node['vector'], dtype=torch.float32)

        tmp_orn = orientation_to_quaternion(node['position'][1])
        tmp_size_and_pos = list(node['size']) + list(node['position'][0]) + tmp_orn
        node_size_and_pos[node_id] = torch.tensor(tmp_size_and_pos, dtype=torch.float32)

        # Legacy orientation normalization has been replaced by orientation_to_quaternion().

    for node in closeToAgent:
        node_close_agent[node] = 1
    g.ndata['close'] = node_close_agent
    g.ndata['feat'] = torch.cat((node_vectors, node_states, node_size_and_pos), 1)

    # print('--' * 20)
    # print(g.ndata['feat'].shape)  # *torch.Size([38, 338])
    # print(node_vectors.shape)  # * torch.Size([38, 300])
    # print(node_states.shape)  # * torch.Size([38, 28])
    # print(node_size_and_pos.shape)  # * torch.Size([38, 10])

    # * -------------------------------
    # g.ndata['0'] = node_vectors
    # g.ndata['1'] = node_states
    # g.ndata['2'] = node_size_and_pos[:, :3]  # * size
    # g.ndata['3'] = node_size_and_pos[:, 3:6]  # * pos
    # g.ndata['4'] = node_size_and_pos[:, 6::]  # * orn

    return g


def convert_datapoint_to_dgl_graph(config, datapoint):
    for i in range(len(datapoint.metrics)):
        if datapoint.actions[i] == 'End':
            dgl_graph = convertToDGLGraph(
                config,
                datapoint.getGraph(i, embeddings=config.embedding)['graph_' + str(i)],
                False,
                -1
            )
    return dgl_graph
