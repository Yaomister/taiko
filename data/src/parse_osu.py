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
