import json
import torch


class EnvironmentConfig:
    def __init__(self, args):
        self.device = args.device if args.device is not None else None
        self.world = args.world
        self.input = args.input
        self.logging = args.logging
        self.speed = args.speed
        self.goal = args.goal
        self.randomize = args.randomize
        self.domain = "home"
        self.display = args.display
        self.add_state_noise = args.state_noise

        self.training = args.training
        self.model_name = args.model_name
        self.exec_type = args.exec_type
        self.split = args.split
        self.globalnode = args.global_node
        self.ignoreNoTool = args.ignoreNoTool
        self.graph_seq_length = 4
        self.goal_representation_type = 2

        self.data_dir = args.data_dir
        self.Aall_path = self.data_dir + "/A_all_restore.pkl"
        self.all_possible_actions_path = self.data_dir + "/all_possible_restore.pkl"
        self.sequence = "seq" in self.training or "action" in self.training
        self.weighted = ("_W" in self.model_name) ^ ("Final" in self.model_name)
        self.embedding = "conceptnet"

        # Scene configuration for Isaac Lab / expff.usd.
        self.scene_usd_path = "/root/Desktop/Collected_exp3/expff.usd"
        self.graph_world_name = "world_expff"
        self.exploration_goal_name = "exploration"

        # Minimal state vocabulary requested by the user.
        self.STATES = [
            "Outside",
            "Inside",
            "Grabbed",
            "Free",
            "Different_Height",
            "Same_Height",
        ]
        self.N_STATES = len(self.STATES)
        self.state2indx = {state: i for i, state in enumerate(self.STATES)}
        self.possibleStates = [self._to_action_state(state) for state in self.STATES]
        self.policy_backend = str(getattr(args, "policy_backend", "isaaclab")).lower()

        self.INVERSE_STATE = {
            "outside": "inside",
            "inside": "outside",
            "grabbed": "free",
            "free": "grabbed",
            "different_height": "same_height",
            "same_height": "different_height",
        }

        self.EDGES = ["Close", "Inside", "On", "Stuck"]
        self.N_EDGES = len(self.EDGES)
        self.edge2idx = {edge: i for i, edge in enumerate(self.EDGES)}

        self.PRETRAINED_VECTOR_SIZE = 300
        self.REDUCED_DIMENSION_SIZE = 64
        self.SIZE_AND_POS_SIZE = 10
        self.features_dim = self.PRETRAINED_VECTOR_SIZE + self.N_STATES + 10
        self.N_TIMESEPS = 2
        self.GRAPH_HIDDEN = 64
        self.NUM_EPOCHS = args.num_epochs
        self.LOGIT_HIDDEN = 32
        self.NUM_GOALS = 8
        self.AUGMENTATION = 1
        self.MODEL_SAVE_PATH = "checkpoints/" + self.domain + "/"

        # TAL object aliases -> USD prim names in expff.usd.
        self.tal_to_usd = {
            "husky": "Mobie_grasper2",
            "cube_red": "Cube",
            "tray": "SmallPallet",
            "big-tray": "BigPallet",
            "bottle_red": "Bottle2",
            "stool": "Stool",
            "table": "table",
        }
        self.usd_to_tal = {v: k for k, v in self.tal_to_usd.items()}
        self.all_objects = list(self.tal_to_usd.keys())
        self.allObjects = self.all_objects
        self.num_objects = len(self.all_objects)
        self.NUMOBJECTS = self.num_objects
        self.object2idx = {obj: i for i, obj in enumerate(self.all_objects)}
        self.idx2object = {i: obj for i, obj in enumerate(self.all_objects)}

        # Object semantics used by the symbolic planner / graph builder.
        self.object_property_map = {
            "husky": ["Robot"],
            "cube_red": ["Movable", "Stickable"],
            "tray": ["Movable", "Surface", "Container"],
            "big-tray": ["Surface", "Container", "Drivable"],
            "bottle_red": ["Movable", "Stickable", "Can_Fuel"],
            "stool": ["Surface", "Can_Lift"],
            "table": ["Surface"],
        }

        self.objects = [
            {
                "name": obj,
                "properties": list(self.object_property_map.get(obj, [])),
                "size": [0.1, 0.1, 0.1],
            }
            for obj in self.all_objects
        ]

        self.property2Objects = {
            "all": list(self.all_objects),
            "Surface": ["tray", "big-tray", "stool", "table"],
            "Container": ["tray", "big-tray"],
            "Movable": ["cube_red", "tray", "bottle_red"],
            "Stickable": ["cube_red", "bottle_red"],
            "Can_Fuel": ["bottle_red"],
            "Drivable": ["big-tray"],
            "Can_Lift": ["stool"],
        }
        self.surfaceAndContainers = list(
            dict.fromkeys(self.property2Objects["Surface"] + self.property2Objects["Container"])
        )

        # Objects that the robot actively manipulates in exploration.
        self.actionable_objects = ["cube_red", "tray", "big-tray", "bottle_red", "stool"]
        self.navigation_targets = [obj for obj in self.all_objects if obj != "husky"]
        self.place_targets = list(dict.fromkeys(self.surfaceAndContainers))
        self.large_objects = {"big-tray", "stool", "table"}

        # 2026-05-30 修改：为在线 moveTo 导航补充接近距离、方向和占地参数，
        # 供 TAL 高层动作解析为 Nav2 可执行目标点。
        self.base_approach_distance = 0.50
        self.pick_approach_distance = 0.32
        self.place_approach_distance = 0.40
        self.push_approach_distance = 0.45
        self.nav_approach_distance_overrides = {
            "tray": 0.36,
            "big-tray": 0.32,
            "stool": 0.30,
            "table": 0.38,
        }
        self.arm_effective_reach_m = 0.40
        self.nav_approach_direction_overrides = {
            "tray": [-1.0, 0.0],
        }
        self.robot_root_yaw_offset = 0.0
        self.lidar_footprint_overrides = {
            "cube_red": [0.06, 0.06],
            "tray": [0.36, 0.24],
            "big-tray": [0.52, 0.38],
            "bottle_red": [0.08, 0.08],
            "stool": [0.42, 0.42],
            "table": [0.90, 0.60],
        }
        self.object_size_overrides = {
            "cube_red": [0.10, 0.10, 0.10],
            "tray": [0.36, 0.24, 0.08],
            "big-tray": [0.52, 0.38, 0.10],
            "bottle_red": [0.08, 0.08, 0.24],
            "stool": [0.42, 0.42, 0.45],
            "table": [0.90, 0.60, 0.50],
        }

        # changeState is removed from the reduced real-robot action space.
        self.object_state_map = {}
        self.hasState = []
        self.all_objects_with_states = []
        self.allStates = {"home": {}, "factory": {}}
        self.initial_object_metrics = {}
        self.usd_metadata = {}

        self.TOOLS2 = ["cube_red"]
        self.TOOLS = self.TOOLS2 + ["no-tool"]
        self.NUMTOOLS = len(self.TOOLS)
        self.goal_jsons = []
        self.new_goal_jsons = []
        self.goalObjects = {}
        self.tools = self.TOOLS2
        self.printable = []
        self.skip = ["husky"]

        # Minimal action set requested by the user.
        self.possibleActions = ["drop", "pick", "moveTo", "pushTo", "pickNplaceAonB"]
        self.num_actions = len(self.possibleActions)
        self.noArgumentActions = []
        self.singleArgumentActions = ["moveTo", "pick", "drop"]
        self.ACTION_ARGS_NUM = {
            "moveTo": 1,
            "pick": 1,
            "drop": 1,
            "pushTo": 2,
            "pickNplaceAonB": 2,
        }

        (
            self.embeddings,
            self.object2vec,
            _,
            _,
            self.tool_vec,
            self.goal2vec,
            self.goalObjects2vec,
        ) = self.compute_constants(self.embedding)
        self.embeddings = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in self.embeddings.items()
        }
        self.tool_vec = self.tool_vec.to(self.device)
        self.goal2vec, self.goalObjects2vec = {}, {}

        self.etypes = ["Close", "Inside", "On", "Stuck"]

    @staticmethod
    def _to_action_state(state_name: str) -> str:
        return state_name.lower()

    @staticmethod
    def _to_canonical_state(state_name: str) -> str:
        tokens = [token for token in state_name.replace("-", "_").split("_") if token]
        return "_".join(token.capitalize() for token in tokens)

    def canonical_state_name(self, state_name):
        if state_name is None:
            return None
        if state_name in self.state2indx:
            return state_name
        normalized = str(state_name).strip().replace("-", "_")
        if not normalized:
            return None
        if normalized in self.state2indx:
            return normalized
        lower_state = normalized.lower()
        for candidate in self.STATES:
            if candidate.lower() == lower_state:
                return candidate
        title_state = self._to_canonical_state(normalized)
        if title_state in self.state2indx:
            return title_state
        return None

    def normalize_state_name(self, state_name):
        canonical = self.canonical_state_name(state_name)
        return canonical.lower() if canonical is not None else None

    def get_object_entry(self, obj_name):
        for obj in self.objects:
            if obj["name"] == obj_name:
                return obj
        return None

    def get_object_properties(self, obj_name):
        obj_entry = self.get_object_entry(obj_name)
        if obj_entry is None:
            return []
        return obj_entry["properties"]

    def compute_constants(self, embedding):
        with open("src/envs/jsons/embeddings/" + embedding + ".vectors") as handle:
            embeddings = json.load(handle)
        object2vec = {obj: embeddings[obj] for obj in self.all_objects}
        tool_vec = torch.tensor([object2vec[i] for i in self.TOOLS2], dtype=torch.float32)
        return embeddings, object2vec, self.object2idx, self.idx2object, tool_vec, None, None
