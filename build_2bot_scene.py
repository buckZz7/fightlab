"""Build a 2-bot MuJoCo scene from g1.xml using proper XML section parsing."""
import os, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_sections(xml):
    """Parse top-level XML sections from inside <mujoco>...</mujoco>."""
    inner = re.search(r'<mujoco[^>]*>(.*)</mujoco>', xml, re.DOTALL).group(1)
    depth = 0; start = 0; sections = []
    for m in re.finditer(r'<(/?)(\w+)[^>]*?(/?)>', inner):
        close, tag, selfclose = m.groups()
        if close:
            depth -= 1
            if depth == 0:
                sections.append(inner[start:m.end()])
                start = m.end()
        elif not selfclose:
            if depth == 0:
                start = m.start()
            depth += 1
        elif depth == 0:
            sections.append(inner[start:m.end()])
            start = m.end()
    return {re.match(r'<(\w+)', s.strip()).group(1): s for s in sections}


def prefix_all_names(text, prefix="r2_"):
    """Prefix all name= attributes in XML."""
    return re.sub(r'name="([^"]+)"', lambda m: f'name="{prefix}{m.group(1)}"', text)


def prefix_joint_refs(text, prefix="r2_"):
    """Prefix all joint= attributes."""
    return re.sub(r'joint="([^"]+)"', lambda m: f'joint="{prefix}{m.group(1)}"', text)


def build_2bot_scene():
    g1_path = os.path.join(SCRIPT_DIR, "g1.xml")
    sections = parse_sections(open(g1_path).read())

    compiler = sections.get("compiler", '<compiler angle="radian" />')
    option = sections.get("option", '<option integrator="implicitfast" timestep="0.005" />')
    defaults = sections.get("default", "")
    assets = sections.get("asset", "")
    worldbody = sections.get("worldbody", "")
    actuators = sections.get("actuator", "")
    contacts = sections.get("contact", "")
    sensors = sections.get("sensor", "")

    # Strip the <worldbody> tags to get inner content
    wb_inner = re.search(r'<worldbody>(.*)</worldbody>', worldbody, re.DOTALL).group(1)
    act_inner = re.search(r'<actuator>(.*)</actuator>', actuators, re.DOTALL).group(1)
    contact_inner = re.search(r'<contact>(.*)</contact>', contacts, re.DOTALL).group(1) if contacts else ""

    # Create r2 versions
    r2_wb = prefix_all_names(wb_inner)
    # Move r2 to x=+0.6, rotate 180 to face r1
    r2_wb = r2_wb.replace(
        'name="r2_pelvis" pos="-0.6 0 0.79"',
        'name="r2_pelvis" pos="0.6 0 0.79" quat="0 0 0 1"'
    )

    r2_act = prefix_all_names(act_inner)
    r2_act = prefix_joint_refs(r2_act)

    r2_contact = prefix_all_names(contact_inner)

    scene = f"""<mujoco model="fightlab_2bot">
  {option}
  {compiler}
  {defaults}
  {assets}
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.05 0.05 0.08 1"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="broadcast" mode="trackcom" pos="-0.15 -4.0 1.2" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <!-- Robot 1 (left, facing +X) -->
    {wb_inner}
    <!-- Robot 2 (right, facing -X) -->
    {r2_wb}
  </worldbody>
  <actuator>
    {act_inner}
    {r2_act}
  </actuator>
  <contact>
    {contact_inner}
    {r2_contact}
  </contact>
</mujoco>
"""
    return scene


if __name__ == "__main__":
    path = os.path.join(SCRIPT_DIR, "scene_2bot.xml")
    with open(path, 'w') as f:
        f.write(build_2bot_scene())
    print(f"Saved {path}")
