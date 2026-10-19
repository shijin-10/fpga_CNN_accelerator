
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

    reg signed [31:0] reference_conv2 [0:3135];
    reg signed [31:0] reference_pool2 [0:783];
    reg signed [31:0] reference_fc_input [0:783];
    reg signed [63:0] reference_scores [0:9];

    integer i;
    integer cycles;
    integer scan_result;
    integer ref_file;
    integer conv2_errors;
    integer pool2_errors;
    integer fc_input_errors;
    integer score_errors;

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

        // Read Conv2 reference outputs as decimal values.
        ref_file = $fopen("conv2_reference.mem", "r");
        if (ref_file == 0) begin
            $display("ERROR: Cannot open conv2_reference.mem");
            $finish;
        end

        for (i = 0; i < 3136; i = i + 1) begin
            scan_result = $fscanf(ref_file, "%d", reference_conv2[i]);
            if (scan_result != 1) begin
                $display("ERROR reading Conv2 reference index %0d", i);
                $fclose(ref_file);
                $finish;
            end
        end
        $fclose(ref_file);

        // Read Pool2 reference outputs.
        ref_file = $fopen("conv2_pool_output.mem", "r");
        if (ref_file == 0) begin
            $display("ERROR: Cannot open conv2_pool_output.mem");
            $finish;
        end

        for (i = 0; i < 784; i = i + 1) begin
            scan_result = $fscanf(ref_file, "%d", reference_pool2[i]);
            if (scan_result != 1) begin
                $display("ERROR reading Pool2 reference index %0d", i);
                $fclose(ref_file);
                $finish;
            end
        end
        $fclose(ref_file);

        // Read expected FC input values.
        ref_file = $fopen("fc_input_requantized.mem", "r");
        if (ref_file == 0) begin
            $display("ERROR: Cannot open fc_input_requantized.mem");
            $finish;
        end

        for (i = 0; i < 784; i = i + 1) begin
            scan_result = $fscanf(ref_file, "%d", reference_fc_input[i]);
            if (scan_result != 1) begin
                $display("ERROR reading FC input reference index %0d", i);
                $fclose(ref_file);
                $finish;
            end
        end
        $fclose(ref_file);

        // Read reference FC scores.
        ref_file = $fopen("fc_output.mem", "r");
        if (ref_file == 0) begin
            $display("ERROR: Cannot open fc_output.mem");
            $finish;
        end

        for (i = 0; i < 10; i = i + 1) begin
            scan_result = $fscanf(ref_file, "%d", reference_scores[i]);
            if (scan_result != 1) begin
                $display("ERROR reading FC score reference %0d", i);
                $fclose(ref_file);
                $finish;
            end
        end
        $fclose(ref_file);

        // Reset and start CNN.
        repeat (5) @(negedge clk);
        rst = 0;

        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        $display("Running complete sequential CNN...");

        cycles = 0;
        while ((done !== 1'b1) && (cycles < 500000)) begin
            @(negedge clk);
            cycles = cycles + 1;
        end

        if (done !== 1'b1) begin
            $display("FAIL: Timeout waiting for CNN completion.");
            $display("Cycles=%0d busy=%b", cycles, busy);
            $finish;
        end

        $display("");
        $display("CNN completed in %0d cycles", cycles);
        $display("Predicted digit: %0d", predicted_digit);

        // Compare Conv2 outputs.
        conv2_errors = 0;
        for (i = 0; i < 3136; i = i + 1) begin
            if ($signed(dut.conv2_memory[i]) !== reference_conv2[i]) begin
                if (conv2_errors < 10)
                    $display("Conv2[%0d]: RTL=%0d REF=%0d",
                        i, $signed(dut.conv2_memory[i]),
                        reference_conv2[i]);
                conv2_errors = conv2_errors + 1;
            end
        end

        // Compare Pool2 outputs.
        pool2_errors = 0;
        for (i = 0; i < 784; i = i + 1) begin
            if ($signed(dut.pool2_memory[i]) !== reference_pool2[i]) begin
                if (pool2_errors < 10)
                    $display("Pool2[%0d]: RTL=%0d REF=%0d",
                        i, $signed(dut.pool2_memory[i]),
                        reference_pool2[i]);
                pool2_errors = pool2_errors + 1;
            end
        end

        // Print a small sample to locate the first divergence.
        $display("");
        $display("First 25 values:");
        for (i = 0; i < 25; i = i + 1) begin
            $display(
                "i=%0d conv2_RTL=%0d conv2_REF=%0d pool2_RTL=%0d pool2_REF=%0d fc_RTL=%0d fc_REF=%0d",
                i,
                $signed(dut.conv2_memory[i]),
                reference_conv2[i],
                $signed(dut.pool2_memory[i]),
                reference_pool2[i],
                $signed(dut.fc_input8[i]),
                reference_fc_input[i]
            );
        end

        // Compare FC inputs.
        fc_input_errors = 0;
        for (i = 0; i < 784; i = i + 1) begin
            if ($signed(dut.fc_input8[i]) !== reference_fc_input[i]) begin
                if (fc_input_errors < 10)
                    $display("FC input[%0d]: RTL=%0d REF=%0d",
                        i, $signed(dut.fc_input8[i]),
                        reference_fc_input[i]);
                fc_input_errors = fc_input_errors + 1;
            end
        end

        // Compare FC scores.
        score_errors = 0;
        for (i = 0; i < 10; i = i + 1) begin
            $display("Digit %0d: RTL=%0d REF=%0d",
                i, scores[i], reference_scores[i]);

            if (scores[i] !== reference_scores[i])
                score_errors = score_errors + 1;
        end

        $display("");
        $display("============== SUMMARY ==============");
        $display("Conv2 mismatches:    %0d / 3136", conv2_errors);
        $display("Pool2 mismatches:    %0d / 784", pool2_errors);
        $display("FC input mismatches: %0d / 784", fc_input_errors);
        $display("Score mismatches:    %0d / 10", score_errors);

        if ((conv2_errors == 0) &&
            (pool2_errors == 0) &&
            (fc_input_errors == 0) &&
            (score_errors == 0))
            $display("PASS: COMPLETE CNN MATCHES REFERENCE");
        else
            $display("FAIL: DIFFERENCES FOUND");

        $display("=====================================");
        $finish;
    end

endmodule

