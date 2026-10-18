
module conv2_16filters (
    input  signed [7:0]  input_data [0:1567],
    input  signed [7:0]  weights    [0:1151],
    input  signed [31:0] biases     [0:15],
    output reg signed [31:0] output_data [0:3135]
);

    integer f, c, y, x, ky, kx;
    integer input_index;
    integer weight_index;
    integer output_index;

    reg signed [31:0] accumulator;
    reg signed [15:0] product;

    always @(*) begin
        for (f = 0; f < 16; f = f + 1) begin
            for (y = 0; y < 14; y = y + 1) begin
                for (x = 0; x < 14; x = x + 1) begin

                    accumulator = biases[f];

                    for (c = 0; c < 8; c = c + 1) begin
                        for (ky = -1; ky <= 1; ky = ky + 1) begin
                            for (kx = -1; kx <= 1; kx = kx + 1) begin

                                if ((y + ky >= 0) &&
                                    (y + ky < 14) &&
                                    (x + kx >= 0) &&
                                    (x + kx < 14)) begin

                                    input_index =
                                        c * 196 +
                                        (y + ky) * 14 +
                                        (x + kx);

                                    weight_index =
                                        f * 72 +
                                        c * 9 +
                                        (ky + 1) * 3 +
                                        (kx + 1);

                                    product =
                                        $signed(input_data[input_index]) *
                                        $signed(weights[weight_index]);

                                    accumulator =
                                        accumulator + product;
                                end
                            end
                        end
                    end

                    output_index = f * 196 + y * 14 + x;

                    // ReLU
                    if (accumulator < 0)
                        output_data[output_index] = 0;
                    else
                        output_data[output_index] = accumulator;

                end
            end
        end
    end

endmodule
