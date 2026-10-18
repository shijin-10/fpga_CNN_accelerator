module conv5x5 (
    input signed [7:0] image [0:24],
    input signed [7:0] kernel [0:8],

    output reg signed [31:0] output_data [0:8]
);

    integer row;
    integer col;
    integer i;

    integer acc;

    integer pixel_index;
    integer kernel_index;


    always @(*) begin

        for (row = 0; row < 3; row = row + 1) begin

            for (col = 0; col < 3; col = col + 1) begin

                acc = 0;

                for (i = 0; i < 3; i = i + 1) begin

                    pixel_index = (row + i) * 5 + col;

                    kernel_index = i * 3;


                    acc = acc +
                          image[pixel_index] *
                          kernel[kernel_index];


                    acc = acc +
                          image[pixel_index + 1] *
                          kernel[kernel_index + 1];


                    acc = acc +
                          image[pixel_index + 2] *
                          kernel[kernel_index + 2];

                end

                output_data[row * 3 + col] = acc;

            end

        end

    end

endmodule
