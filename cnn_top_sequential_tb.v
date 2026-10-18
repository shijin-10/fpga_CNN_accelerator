
`timescale 1ns/1ps

module cnn_top_sequential_tb;

    reg clk = 0;
    reg rst = 1;
    reg start = 0;

    reg signed [7:0] image [0:783];
    reg signed [7:0] conv1_weights [0:71];
    reg signed [31:0] conv1_biases [0:7];
    reg signed [7:0] conv2_weights [0:1151];
    reg signed [31:0] conv2_biases [0:15];
    reg signed [7:0] fc_weights [0:7839];
    reg signed [31:0] fc_biases [0:9];

    wire busy;
    wire done;
    wire [3:0] predicted_digit;
    wire signed [63:0] scores [0:9];

    reg signed [63:0] reference_scores [0:9];
    reg signed [31:0] expected_fc_input [0:783];

    integer i;
    integer cycles;
    integer score_errors;
    integer input_errors;
    integer ref_file;
    integer scan_result;

    cnn_top_sequential dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .image(image),
        .conv1_weights(conv1_weights),
        .conv1_biases(conv1_biases),
        .conv2_weights(conv2_weights),
        .conv2_biases(conv2_biases),
        .fc_weights(fc_weights),
        .fc_biases(fc_biases),
        .busy(busy),
        .done(done),
        .predicted_digit(predicted_digit),
        .scores(scores)
    );

    always #5 clk = ~clk;

    initial begin
        $display("Loading CNN memory files...");

        $readmemh("mnist_image.mem", image);
        $readmemh("conv1_weights.mem", conv1_weights);
        $readmemh("conv1_bias_int32.mem", conv1_biases);
        $readmemh("conv2_weights.mem", conv2_weights);
        $readmemh("conv2_bias_int32.mem", conv2_biases);
        $readmemh("fc_weights.mem", fc_weights);
        $readmemh("fc_bias_int32.mem", fc_biases);

        // Read expected FC inputs as DECIMAL values.
        ref_file = $fopen("fc_input_requantized.mem", "r");

        if (ref_file == 0) begin
            $display("ERROR: Cannot open fc_input_requantized.mem");
            $finish;
        end

        for (i = 0; i < 784; i = i + 1) begin
            scan_result = $fscanf(ref_file, "%d", expected_fc_input[i]);

            if (scan_result != 1) begin
                $display("ERROR: Failed to read FC input %0d", i);
                $fclose(ref_file);
                $finish;
            end
        end

        $fclose(ref_file);

        // Read reference FC scores as DECIMAL values.
        ref_file = $fopen("fc_output.mem", "r");

        if (ref_file == 0) begin
            $display("ERROR: Cannot open fc_output.mem");
            $finish;
        end

        for (i = 0; i < 10; i = i + 1) begin
            scan_result = $fscanf(ref_file, "%d", reference_scores[i]);

            if (scan_result != 1) begin
                $display("ERROR: Failed to read reference score %0d", i);
                $fclose(ref_file);
                $finish;
            end
        end

        $fclose(ref_file);

        // Reset the CNN.
        repeat (5) @(negedge clk);
        rst = 0;

        // Start the CNN for one clock cycle.
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        $display("Running complete sequential CNN...");

        cycles = 0;

        // Check done on the falling edge to avoid missing its pulse.
        while ((done !== 1'b1) && (cycles < 500000)) begin
            @(negedge clk);
            cycles = cycles + 1;
        end

        if (done !== 1'b1) begin
            $display("FAIL: Timeout waiting for CNN completion.");
            $display("Cycles: %0d, busy=%b", cycles, busy);
            $finish;
        end

        $display("");
        $display("CNN completed in %0d cycles", cycles);
        $display("Predicted digit: %0d", predicted_digit);

        // Compare final requantized FC inputs.
        $display("");
        $display("FC input comparison:");

        input_errors = 0;

        for (i = 0; i < 784; i = i + 1) begin
            if ($signed(dut.fc_input8[i]) !== expected_fc_input[i]) begin
                if (input_errors < 10) begin
                    $display(
                        "Mismatch at %0d: RTL=%0d Reference=%0d",
                        i,
                        $signed(dut.fc_input8[i]),
                        expected_fc_input[i]
                    );
                end

                input_errors = input_errors + 1;
            end
        end

        $display("FC input mismatches: %0d / 784", input_errors);

        // Compare all ten final FC scores.
        $display("");
        $display("Score comparison:");

        score_errors = 0;

        for (i = 0; i < 10; i = i + 1) begin
            $display(
                "Digit %0d: RTL=%0d Reference=%0d",
                i, scores[i], reference_scores[i]
            );

            if (scores[i] !== reference_scores[i]) begin
                score_errors = score_errors + 1;
                $display("  MISMATCH at digit %0d", i);
            end
        end

        $display("");
        $display("FC input mismatches: %0d / 784", input_errors);
        $display("Score mismatches: %0d / 10", score_errors);

        if ((input_errors == 0) && (score_errors == 0))
            $display("PASS: COMPLETE CNN MATCHES REFERENCE");
        else
            $display("FAIL: DIFFERENCES FOUND");

        $finish;
    end

endmodule
