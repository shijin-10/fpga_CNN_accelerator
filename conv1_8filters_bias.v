module conv1_8filters_bias (
    input signed [7:0] image [0:783],
    input signed [7:0] weights [0:71],
    input signed [31:0] biases [0:7],
    output reg signed [31:0] output_data [0:6271]
);

    integer row, col, filter, i, j;
    integer acc;
    integer image_index, weight_index, output_index;

    always @(*) begin
        for (filter = 0; filter < 8; filter = filter + 1) begin
            for (row = 0; row < 28; row = row + 1) begin
                for (col = 0; col < 28; col = col + 1) begin

                    // Start with this filter's INT32 bias
                    acc = biases[filter];

                    // 3x3 convolution, padding = 1
                    for (i = 0; i < 3; i = i + 1) begin
                        for (j = 0; j < 3; j = j + 1) begin
                            if ((row+i-1 >= 0) &&
                                (row+i-1 < 28) &&
                                (col+j-1 >= 0) &&
                                (col+j-1 < 28)) begin

                                image_index = (row+i-1)*28 + (col+j-1);
                                weight_index = filter*9 + i*3 + j;

                                acc = acc +
                                      image[image_index] *
                                      weights[weight_index];
                            end
                        end
                    end

                    // ReLU
                    if (acc < 0)
                        acc = 0;

                    output_index = filter*784 + row*28 + col;
                    output_data[output_index] = acc;
                end
            end
        end
    end

endmodule
