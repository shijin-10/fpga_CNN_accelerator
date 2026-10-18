
`timescale 1ns/1ps

module cnn_top_debug_tb;

    reg signed [7:0] image [0:783];
    reg signed [7:0] conv1_weights [0:71];
    reg signed [31:0] conv1_biases [0:7];

    reg signed [7:0] conv2_weights [0:1151];
    reg signed [31:0] conv2_biases [0:15];

    reg signed [7:0] fc_weights [0:7839];
    reg signed [31:0] fc_biases [0:9];

    wire signed [63:0] scores [0:9];

    integer i;
    integer predicted_digit;
    reg signed [63:0] best_score;

    cnn_top dut (
        .image(image),
        .conv1_weights(conv1_weights),
        .conv1_biases(conv1_biases),
        .conv2_weights(conv2_weights),
        .conv2_biases(conv2_biases),
        .fc_weights(fc_weights),
        .fc_biases(fc_biases),
        .scores(scores)
    );

    initial begin
        $display("Loading CNN parameters...");

        $readmemh("mnist_image.mem", image);
        $readmemh("conv1_weights.mem", conv1_weights);
        $readmemh("conv1_bias_int32.mem", conv1_biases);

        $readmemh("conv2_weights.mem", conv2_weights);
        $readmemh("conv2_bias_int32.mem", conv2_biases);

        $readmemh("fc_weights.mem", fc_weights);
        $readmemh("fc_bias_int32.mem", fc_biases);

        $display("Running integrated CNN pipeline...");

        // Allow the combinational pipeline to settle.
        #100;

        // ------------------------------------------
        // DEBUG 1: Conv1 and MaxPool1
        // ------------------------------------------
        $display("");
        $display("========== CONV1 / POOL1 DEBUG ==========");
        $display("pool1_out[0]   = %0d", dut.pool1_out[0]);
        $display("pool1_out[1]   = %0d", dut.pool1_out[1]);
        $display("pool1_out[100] = %0d", dut.pool1_out[100]);
        $display("pool1_out[500] = %0d", dut.pool1_out[500]);

        // ------------------------------------------
        // DEBUG 2: Requantized Conv2 inputs
        // ------------------------------------------
        $display("");
        $display("========== CONV2 INPUT DEBUG ==========");
        $display("conv2_in[0]   = %0d", dut.conv2_in[0]);
        $display("conv2_in[1]   = %0d", dut.conv2_in[1]);
        $display("conv2_in[100] = %0d", dut.conv2_in[100]);
        $display("conv2_in[500] = %0d", dut.conv2_in[500]);

        // ------------------------------------------
        // DEBUG 3: Conv2 and MaxPool2
        // ------------------------------------------
        $display("");
        $display("========== CONV2 / POOL2 DEBUG ==========");
        $display("conv2_out[0]   = %0d", dut.conv2_out[0]);
        $display("conv2_out[1]   = %0d", dut.conv2_out[1]);
        $display("conv2_out[100] = %0d", dut.conv2_out[100]);

        $display("pool2_out[0]   = %0d", dut.pool2_out[0]);
        $display("pool2_out[1]   = %0d", dut.pool2_out[1]);
        $display("pool2_out[100] = %0d", dut.pool2_out[100]);

        // ------------------------------------------
        // DEBUG 4: Requantized FC inputs
        // ------------------------------------------
        $display("");
        $display("========== FC INPUT DEBUG ==========");
        $display("fc_input8[0]   = %0d", dut.fc_input8[0]);
        $display("fc_input8[1]   = %0d", dut.fc_input8[1]);
        $display("fc_input8[100] = %0d", dut.fc_input8[100]);
        $display("fc_input8[500] = %0d", dut.fc_input8[500]);

        // ------------------------------------------
        // DEBUG 5: Final classification
        // ------------------------------------------
        predicted_digit = 0;
        best_score = scores[0];

        $display("");
        $display("========== CNN OUTPUT SCORES ==========");

        for (i = 0; i < 10; i = i + 1) begin
            $display("Digit %0d: %0d", i, scores[i]);

            if (scores[i] > best_score) begin
                best_score = scores[i];
                predicted_digit = i;
            end
        end

        $display("=======================================");
        $display("Predicted digit: %0d", predicted_digit);
        $display("=======================================");

        $finish;
    end

endmodule
