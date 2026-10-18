
def convert_file(source, destination):
    with open(source) as f:
        values = [int(line.strip()) for line in f if line.strip()]

    with open(destination, "w") as f:
        for value in values:
            f.write(f"{value & 0xFFFFFFFF:08x}\n")

    print(f"{source}: converted {len(values)} values")

convert_file(
    "conv1_pool_output.mem",
    "conv1_pool_output_hex.mem"
)

convert_file(
    "conv2_pool_output.mem",
    "conv2_pool_output_hex.mem"
)
