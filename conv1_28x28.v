module conv1_28x28 (
    input signed [7:0] image [0:783],
    input signed [7:0] kernel [0:8],
    output reg signed [31:0] output_data [0:783]
);

    integer row;
    integer col;
    integer i;
    integer j;
    integer acc;
    integer image_index;
    integer kernel_index;

    always @(*) begin

        // Clear output
        for (i = 0; i < 784; i = i + 1)
            output_data[i] = 0;

        // 3x3 convolution with padding = 1
        for (row = 0; row < 28; row = row + 1) begin

            for (col = 0; col < 28; col = col + 1) begin

                acc = 0;

                for (i = 0; i < 3; i = i + 1) begin

                    for (j = 0; j < 3; j = j + 1) begin

                        // Check whether pixel is inside image
                        if ((row + i - 1 >= 0) &&
                            (row + i - 1 < 28) &&
                            (col + j - 1 >= 0) &&
                            (col + j - 1 < 28)) begin

                            image_index =
                                (row + i - 1) * 28 +
                                (col + j - 1);

                            kernel_index = i * 3 + j;

                            acc = acc +
                                  image[image_index] *
                                  kernel[kernel_index];
                        end

                    end
                end

                output_data[row * 28 + col] = acc;

            end
        end
    end

endmodule
