import ast
from pathlib import Path


HARDWARE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "hardware"
ENTRYPOINTS = [
    HARDWARE_DIR / "teleop_G1D_hardware_omnihand.py",
    HARDWARE_DIR / "teleop_recorder.py",
]


def _main_definition(path: Path) -> ast.FunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def test_both_entrypoints_keep_omni_as_the_default_backend():
    for path in ENTRYPOINTS:
        main = _main_definition(path)
        defaults = dict(
            zip(
                [arg.arg for arg in main.args.args[-len(main.args.defaults) :]],
                main.args.defaults,
                strict=True,
            )
        )
        assert isinstance(defaults["gripper_type"], ast.Name)
        assert defaults["gripper_type"].id == "OMNI_GRIPPER"


def test_both_entrypoints_keep_the_existing_omni_urdf_default():
    for path in ENTRYPOINTS:
        main = _main_definition(path)
        robot_urdf_default = main.args.defaults[0]
        assert isinstance(robot_urdf_default, ast.Call)
        assert isinstance(robot_urdf_default.args[-1], ast.Constant)
        assert robot_urdf_default.args[-1].value == "unitree/g1/g1d_dual_arm_omni.urdf"


def test_both_entrypoints_have_explicit_builtin_branch():
    for path in ENTRYPOINTS:
        source = path.read_text(encoding="utf-8")
        assert "UNITREE_BUILTIN_GRIPPER" in source
        assert "enable_internal_gripper=True" in source
        assert "UnitreeBuiltinGripperAdapter" in source


def _builtin_controller_call(path: Path) -> ast.Call:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(module):
        if not isinstance(node, ast.If):
            continue
        if "UNITREE_BUILTIN_GRIPPER" not in ast.unparse(node.test):
            continue
        branch = ast.Module(body=node.body, type_ignores=[])
        for child in ast.walk(branch):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "G1_29_ArmController"
            ):
                return child
    raise AssertionError(f"built-in controller call not found in {path}")


def test_builtin_entrypoints_use_verified_lowcmd_transport():
    for path in ENTRYPOINTS:
        call = _builtin_controller_call(path)
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords["motion_mode"], ast.Constant)
        assert keywords["motion_mode"].value is False
        assert isinstance(keywords["enable_internal_gripper"], ast.Constant)
        assert keywords["enable_internal_gripper"].value is True


def test_builtin_teleop_holds_current_arms_before_first_vr_target():
    call = _builtin_controller_call(ENTRYPOINTS[0])
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["hold_current_arm_on_init"], ast.Constant)
    assert keywords["hold_current_arm_on_init"].value is True


def test_teleop_integrates_images_only_in_explicit_builtin_branch():
    source = ENTRYPOINTS[0].read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(ENTRYPOINTS[0]))
    image_init_call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_init_integrated_images"
    )
    parent_if = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.If)
        and any(child is image_init_call for child in ast.walk(node))
    )
    condition = ast.unparse(parent_if.test)
    assert "self.real_robot" in condition
    assert "self.gripper_type == UNITREE_BUILTIN_GRIPPER" in condition


def test_teleop_keeps_omni_gripper_update_on_original_base_implementation():
    source = ENTRYPOINTS[0].read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(ENTRYPOINTS[0]))
    update_method = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_update_gripper_target"
    )
    first_if = next(node for node in update_method.body if isinstance(node, ast.If))
    assert ast.unparse(first_if.test) == "self.gripper_type == OMNI_GRIPPER"
    assert "super()._update_gripper_target()" in ast.unparse(first_if)


def test_teleop_keeps_original_omni_ax_home_calls():
    source = ENTRYPOINTS[0].read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(ENTRYPOINTS[0]))
    builtin_if = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.gripper_type == UNITREE_BUILTIN_GRIPPER"
        and "command_and_wait_for_dual_arm_home" in ast.unparse(node)
        and "A+X arm reset failed" in ast.unparse(node)
    )
    omni_branch = ast.unparse(ast.Module(body=builtin_if.orelse, type_ignores=[]))
    assert "self.arm.ctrl_dual_arm_go_home()" in omni_branch
    assert "self._reset()" in omni_branch
    assert "command_and_wait_for_dual_arm_home" not in omni_branch


def test_teleop_builtin_ax_uses_shared_recorder_initial_target():
    source = ENTRYPOINTS[0].read_text(encoding="utf-8")
    assert "arm_target=self.builtin_initial_q_target" in source
    assert "unitree_builtin_initial_q_target()" in source


def test_recorder_keeps_omni_gripper_update_on_original_base_implementation():
    source = ENTRYPOINTS[1].read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(ENTRYPOINTS[1]))
    update_method = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_update_gripper_target"
    )
    first_if = next(node for node in update_method.body if isinstance(node, ast.If))
    assert ast.unparse(first_if.test) == "self.gripper_type == OMNI_GRIPPER"
    assert "super()._update_gripper_target()" in ast.unparse(first_if)


def test_recorder_applies_reliable_ax_home_only_to_builtin_branch():
    source = ENTRYPOINTS[1].read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(ENTRYPOINTS[1]))
    reliable_home_call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "command_and_wait_for_dual_arm_home"
    )
    parent_if = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.gripper_type == UNITREE_BUILTIN_GRIPPER"
        and any(child is reliable_home_call for child in ast.walk(node))
    )
    assert ast.unparse(parent_if.test) == (
        "self.gripper_type == UNITREE_BUILTIN_GRIPPER"
    )


def test_recorder_builtin_ax_is_time_based_while_omni_keeps_frame_gate():
    source = ENTRYPOINTS[1].read_text(encoding="utf-8")
    assert "HoldDurationGate(BUILTIN_AX_HOLD_S)" in source
    assert "builtin_ax_triggered = self.builtin_ax_gate.update(chord_pressed)" in source
    assert "else self.reset_flag >= RESET_FLAG_THRESHOLD" in source


def test_builtin_recorder_keeps_its_existing_explicit_initial_arm_target():
    call = _builtin_controller_call(ENTRYPOINTS[1])
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    target = keywords["initial_q_target"]
    assert isinstance(target, ast.Attribute)
    assert isinstance(target.value, ast.Name)
    assert target.value.id == "self"
    assert target.attr == "g_target"


def _controller_init(path: Path) -> ast.FunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    controller_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "G1DTeleopController"
    )
    return next(
        node
        for node in controller_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


def test_builtin_startup_waits_for_the_same_target_then_syncs_ik():
    expected_targets = {
        ENTRYPOINTS[0]: "self.builtin_initial_q_target",
        ENTRYPOINTS[1]: "self.g_target",
    }
    for path, expected_target in expected_targets.items():
        init = _controller_init(path)
        reliable_home_calls = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "command_and_wait_for_dual_arm_home"
        ]
        assert len(reliable_home_calls) == 1
        keywords = {
            keyword.arg: keyword.value
            for keyword in reliable_home_calls[0].keywords
        }
        assert ast.unparse(keywords["arm_target"]) == expected_target

        ik_sync_calls = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_reset_builtin_placo_arms_to_target"
        ]
        assert any(
            call.lineno > reliable_home_calls[0].lineno for call in ik_sync_calls
        )
