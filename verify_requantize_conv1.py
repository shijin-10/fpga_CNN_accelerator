def read_hex_signed8(path):
    values = []

    with open(path) as f:
        for line in f:
            s = line.strip()

            if s:
                v = int(s, 16)

                if v >= 128:
                    v -= 256

                values.append(v)

    return values


with open("conv2_input_requantized.mem") as f:
    rtl = [int(line.strip()) for line in f if line.strip()]

expected = read_hex_signed8("conv2_input.mem")

mismatches = [
    (i, expected[i], rtl[i])
    for i in range(min(len(expected), len(rtl)))
    if expected[i] != rtl[i]
]

print("Expected values:", len(expected))
print("RTL values:", len(rtl))
print("Mismatches:", len(mismatches))

if mismatches:
    print("First mismatches:", mismatches[:10])
else:
    print("SUCCESS: Conv1 requantization matches the exported Conv2 input!")
