module conv1_relu_28x28_tb;

    reg signed [7:0] image [0:783];
    reg signed [7:0] kernel [0:8];

    wire signed [31:0] output_data [0:783];

    integer i;
    integer row;
    integer col;

    conv1_relu_28x28 dut (
        .image(image),
        .kernel(kernel),
        .output_data(output_data)
    );

    initial begin

        // Same 28x28 test image as before
        for (i = 0; i < 784; i = i + 1)
            image[i] = i % 128;

        // Load trained Conv1 weights
        $readmemh("conv1_weights.mem", kernel);

        #10;

        $display("");
        $display("Conv1 + ReLU - First 5x5 Outputs:");
        $display("");

        for (row = 0; row < 5; row = row + 1) begin

            for (col = 0; col < 5; col = col + 1)
                $write("%7d ", output_data[row * 28 + col]);

            $display("");

        end

        $display("");
        $display("Selected outputs:");

        $display("Output[0]  = %0d", output_data[0]);
        $display("Output[1]  = %0d", output_data[1]);
        $display("Output[2]  = %0d", output_data[2]);
        $display("Output[28] = %0d", output_data[28]);
        $display("Output[29] = %0d", output_data[29]);

        $finish;

    end

endmodule
