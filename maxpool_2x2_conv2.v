
module maxpool_2x2_conv2 (
    input signed [31:0] input_data [0:3135],
    output reg signed [31:0] output_data [0:783]
);

    integer f, y, x;
    integer input_index;
    integer output_index;

    reg signed [31:0] a, b, c, d;
    reg signed [31:0] maximum;

    always @(*) begin
        for (f = 0; f < 16; f = f + 1) begin
            for (y = 0; y < 7; y = y + 1) begin
                for (x = 0; x < 7; x = x + 1) begin

                    input_index = f * 196 + (2*y) * 14 + (2*x);

                    a = input_data[input_index];
                    b = input_data[input_index + 1];
                    c = input_data[input_index + 14];
                    d = input_data[input_index + 15];

                    maximum = a;
                    if (b > maximum) maximum = b;
                    if (c > maximum) maximum = c;
                    if (d > maximum) maximum = d;

                    output_index = f * 49 + y * 7 + x;
                    output_data[output_index] = maximum;
                end
            end
        end
    end

endmodule
