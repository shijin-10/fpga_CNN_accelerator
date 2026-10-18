module conv1_28x28_tb;

    reg signed [7:0] image [0:783];
    reg signed [7:0] kernel [0:8];

    wire signed [31:0] output_data [0:783];

    integer i;
    integer row;
    integer col;

    conv1_28x28 dut (
        .image(image),
        .kernel(kernel),
        .output_data(output_data)
    );

    initial begin

        // ------------------------------------------------
        // Create a simple 28x28 test image
        // Pixel values = 0,1,2,3,...,127,0,1,...
        // ------------------------------------------------
        for (i = 0; i < 784; i = i + 1)
            image[i] = i % 128;

        // ------------------------------------------------
        // Load first trained Conv1 filter
        // ------------------------------------------------
        $readmemh("conv1_weights.mem", kernel);

        #10;

        // ------------------------------------------------
        // Display first 5x5 region of output
        // ------------------------------------------------
        $display("");
        $display("Conv1 Filter 0 - First 5x5 Outputs:");
        $display("");

        for (row = 0; row < 5; row = row + 1) begin
            for (col = 0; col < 5; col = col + 1) begin
                $write("%7d ", output_data[row * 28 + col]);
            end
            $display("");
        end

        // ------------------------------------------------
        // Display some individual outputs
        // ------------------------------------------------
        $display("");
        $display("Selected outputs:");
        $display("Output[0]   = %0d", output_data[0]);
        $display("Output[1]   = %0d", output_data[1]);
        $display("Output[28]  = %0d", output_data[28]);
        $display("Output[29]  = %0d", output_data[29]);
        $display("Output[392] = %0d", output_data[392]);

        $finish;

    end

endmodule
