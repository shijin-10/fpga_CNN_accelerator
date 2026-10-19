
module maxpool_2x2 (
    input signed [31:0] input_data [0:6271],
    output reg signed [31:0] output_data [0:1567]
);

    integer filter;
    integer row;
    integer col;
    integer input_index;
    integer output_index;

    reg signed [31:0] a;
    reg signed [31:0] b;
    reg signed [31:0] c;
    reg signed [31:0] d;
    reg signed [31:0] max_value;

    always @(*) begin
        for (filter = 0; filter < 8; filter = filter + 1) begin
            for (row = 0; row < 14; row = row + 1) begin
                for (col = 0; col < 14; col = col + 1) begin

                    // Top-left input position of the 2x2 window.
                    input_index =
                        filter * 784 +
                        (row * 2) * 28 +
                        (col * 2);

                    // Read the four values.
                    a = input_data[input_index];
                    b = input_data[input_index + 1];
                    c = input_data[input_index + 28];
                    d = input_data[input_index + 29];

                    // Find the maximum.
                    max_value = a;

                    if (b > max_value)
                        max_value = b;

                    if (c > max_value)
                        max_value = c;

                    if (d > max_value)
                        max_value = d;

                    // ReLU: clamp negative activations to zero.
                    if (max_value < 0)
                        max_value = 0;

                    // Store in filter-major order.
                    output_index =
                        filter * 196 +
                        row * 14 +
                        col;

                    output_data[output_index] = max_value;

                end
            end
        end
    end

endmodule
