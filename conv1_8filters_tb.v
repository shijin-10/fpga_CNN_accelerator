module conv1_8filters_tb;

    reg signed [7:0] image [0:783];
    reg signed [7:0] weights [0:71];

    wire signed [31:0] output_data [0:6271];

    integer i;
    integer f;

    conv1_8filters dut (
        .image(image),
        .weights(weights),
        .output_data(output_data)
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
        $display(" Conv1 - 8 Filters + ReLU");
        $display("======================================");

        // Display first output pixel of every filter
        for (f = 0; f < 8; f = f + 1) begin
            $display("Filter %0d, Output[0] = %0d",
                     f,
                     output_data[f * 784]);
        end

        $display("");
        $display("First 3x3 outputs of each filter:");
        $display("");

        for (f = 0; f < 8; f = f + 1) begin

            $display("Filter %0d:", f);

            $display("%7d %7d %7d",
                     output_data[f*784 + 0],
                     output_data[f*784 + 1],
                     output_data[f*784 + 2]);

            $display("%7d %7d %7d",
                     output_data[f*784 + 28],
                     output_data[f*784 + 29],
                     output_data[f*784 + 30]);

            $display("%7d %7d %7d",
                     output_data[f*784 + 56],
                     output_data[f*784 + 57],
                     output_data[f*784 + 58]);

            $display("");

        end

        $finish;

    end

endmodule
