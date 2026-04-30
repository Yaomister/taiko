def parse_osu(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        sections = {}
        current = None
        hit_object_lines = []
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                sections[current] = {}
            elif current == "HitObjects":
                if line:
                    hit_object_lines.append(line)
            elif current and ":" in line:
                key, _, value = line.partition(":")
                sections[current][key.strip()] = value.strip()

    mode    = int(sections.get("General", {}).get("Mode", -1))
    version = sections.get("Metadata", {}).get("Version", "")

    if mode != 0:
        raise ValueError(f"Unsupported mode: {mode}")

    hit_objects = []
    for raw in hit_object_lines:
        parts = raw.split(",")
        type_bits = int(parts[3])
        if type_bits & 1:
            obj_type = "circle"
        elif type_bits & 2:
            obj_type = "slider"
            curve_str = parts[5] if len(parts) > 5 else ""
            curve_parts = curve_str.split("|")
            curve_type = curve_parts[0] if curve_parts else "L"
            cp_dx, cp_dy = 0.0, 0.0
            if len(curve_parts) > 1:
                cp = curve_parts[1].split(":")
                cp_dx = (int(cp[0]) - int(parts[0])) / 512
                cp_dy = (int(cp[1]) - int(parts[1])) / 384
            length = float(parts[7]) if len(parts) > 7 else 100.0
        elif type_bits & 8:
            obj_type = "spinner"
        else:
            continue
        hit_objects.append({
            "x": int(parts[0]) / 512,
            "y": int(parts[1]) / 384,
            "time_ms": int(parts[2]),
            "type": obj_type,
        })

    return {
        "version": version,
        "hit_objects": hit_objects,
    }
