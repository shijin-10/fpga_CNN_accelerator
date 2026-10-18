def read_hex_signed32(path):
    values = []

    with open(path) as f:
        for line in f:
            s = line.strip()

            if s:
                v = int(s, 16)

                if v >= 2**31:
                    v -= 2**32

                values.append(v)

    return values


with open("fc_input_requantized.mem") as f:
    rtl = [int(line.strip()) for line in f if line.strip()]

expected = read_hex_signed32("fc_input.mem")

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
    print("SUCCESS: Conv2 requantization matches the exported FC input!")
