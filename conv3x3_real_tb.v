`timescale 1ns/1ps

module conv3x3_real_tb;

    reg signed [7:0] pixels [0:8];
    reg signed [7:0] weights [0:8];

    integer i;
    integer acc;


    initial begin

        // Load the first 9 weights
        // from the first trained Conv1 filter

        $readmemh(
            "conv1_weights.mem",
            weights
        );

        // Load test pixels

        $readmemh(
            "test_pixels.mem",
            pixels
        );


        // Calculate convolution

        acc = 0;

        for (i = 0; i < 9; i = i + 1) begin

            acc = acc +
                  pixels[i] * weights[i];

        end


        $display("");
        $display("==============================");
        $display("REAL CONV1 FILTER TEST");
        $display("==============================");

        for (i = 0; i < 9; i = i + 1) begin

            $display(
                "Pixel[%0d] = %d   Weight[%0d] = %d",
                i,
                pixels[i],
                i,
                weights[i]
            );

        end

        $display("");
        $display("Accumulator = %d", acc);
        $display("==============================");

        $finish;

    end

endmodule
