module conv1_pool_tb;

    reg signed [7:0] image [0:783];
    reg signed [7:0] weights [0:71];

    wire signed [31:0] conv_output [0:6271];
    wire signed [31:0] pool_output [0:1567];

    integer i;
    integer f;

    // Conv1 + ReLU
    conv1_8filters conv1 (
        .image(image),
        .weights(weights),
        .output_data(conv_output)
    );

    // MaxPool
    maxpool_2x2 pool (
        .input_data(conv_output),
        .output_data(pool_output)
    );

    initial begin

        // Same test image
        for (i = 0; i < 784; i = i + 1)
            image[i] = i % 128;

        // Load all 72 trained Conv1 weights
        $readmemh("conv1_weights.mem", weights);

        #10;

        $display("");
        $display("======================================");
        $display(" Conv1 + ReLU + MaxPool");
        $display("======================================");

        // Show first pooled value of each filter
        for (f = 0; f < 8; f = f + 1) begin
            $display("Filter %0d, Pool[0] = %0d",
                     f,
                     pool_output[f * 196]);
        end

        $display("");
        $display("First 3x3 pooled outputs of Filter 0:");
        $display("");

        $display("%7d %7d %7d",
                 pool_output[0],
                 pool_output[1],
                 pool_output[2]);

        $display("%7d %7d %7d",
                 pool_output[14],
                 pool_output[15],
                 pool_output[16]);

        $display("%7d %7d %7d",
                 pool_output[28],
                 pool_output[29],
                 pool_output[30]);

        $finish;

    end

endmodule
