module conv5x5_real_tb;

    reg signed [7:0] image [0:24];
    reg signed [7:0] kernel [0:8];

    wire signed [31:0] output_data [0:8];

    integer i;

    conv5x5 dut (
        .image(image),
        .kernel(kernel),
        .output_data(output_data)
    );

    initial begin

        // 5x5 test image
        image[0]  = 1;
        image[1]  = 2;
        image[2]  = 3;
        image[3]  = 4;
        image[4]  = 5;

        image[5]  = 6;
        image[6]  = 7;
        image[7]  = 8;
        image[8]  = 9;
        image[9]  = 10;

        image[10] = 11;
        image[11] = 12;
        image[12] = 13;
        image[13] = 14;
        image[14] = 15;

        image[15] = 16;
        image[16] = 17;
        image[17] = 18;
        image[18] = 19;
        image[19] = 20;

        image[20] = 21;
        image[21] = 22;
        image[22] = 23;
        image[23] = 24;
        image[24] = 25;

        // Load all 72 trained Conv1 weights
        // First 9 values = Filter 0
        $readmemh("conv1_weights.mem", kernel);

        #10;

        $display("Conv1 Filter 0:");
        for (i = 0; i < 9; i = i + 1)
            $display("Weight[%0d] = %0d", i, kernel[i]);

        $display("");
        $display("Convolution outputs:");

        for (i = 0; i < 9; i = i + 1)
            $display("Output[%0d] = %0d", i, output_data[i]);

        $finish;
    end

endmodule
