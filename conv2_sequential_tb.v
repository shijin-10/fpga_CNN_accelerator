
`timescale 1ns/1ps

module conv2_sequential_tb;

    reg clk = 0;
    reg rst = 1;
    reg start = 0;

    reg signed [7:0] input_memory [0:1567];
    reg signed [7:0] weight_memory [0:1151];
    reg signed [31:0] bias_memory [0:15];

    wire signed [7:0] input_data;
    wire signed [7:0] weight_data;
    wire signed [31:0] bias_data;

    wire [10:0] input_address;
    wire [10:0] weight_address;
    wire [3:0] bias_address;

    wire [11:0] output_address;
    wire signed [31:0] output_value;

    wire busy;
    wire done;

    reg signed [31:0] reference [0:3135];

    integer i;
    integer errors;
    integer checked;
    integer cycles;
    integer ref_file;
    integer scan_result;

    assign input_data = input_memory[input_address];
    assign weight_data = weight_memory[weight_address];
    assign bias_data = bias_memory[bias_address];

    conv2_sequential dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .input_data(input_data),
        .weight_data(weight_data),
        .bias_data(bias_data),
        .input_address(input_address),
        .weight_address(weight_address),
        .bias_address(bias_address),
        .output_address(output_address),
        .output_value(output_value),
        .busy(busy),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        $display("Loading input, weights, and biases...");

        $readmemh("conv2_input.mem", input_memory);
        $readmemh("conv2_weights.mem", weight_memory);
        $readmemh("conv2_bias_int32.mem", bias_memory);

        // Read decimal reference values, not hexadecimal.
        ref_file = $fopen("conv2_reference.mem", "r");

        if (ref_file == 0) begin
            $display("ERROR: Cannot open conv2_reference.mem");
            $finish;
        end

        for (i = 0; i < 3136; i = i + 1) begin
            scan_result = $fscanf(
                ref_file, "%d", reference[i]
            );

            if (scan_result != 1) begin
                $display(
                    "ERROR: Could not read reference value %0d",
                    i
                );
                $fclose(ref_file);
                $finish;
            end
        end

        $fclose(ref_file);

        $display("Reference loaded as decimal.");
        errors = 0;
        checked = 0;
        cycles = 0;

        repeat (2) @(negedge clk);
        rst = 0;

        start = 1;
        @(negedge clk);
        start = 0;

        while (!done && cycles < 300000) begin
            @(negedge clk);

            if (dut.state == 2) begin
                if (output_value !== reference[output_address]) begin
                    if (errors < 20) begin
                        $display(
                            "MISMATCH addr=%0d got=%0d expected=%0d",
                            output_address,
                            output_value,
                            reference[output_address]
                        );
                    end

                    errors = errors + 1;
                end

                checked = checked + 1;
            end

            cycles = cycles + 1;
        end

        $display("----------------------------------------");
        $display("Outputs checked : %0d", checked);
        $display("Errors          : %0d", errors);
        $display("Cycles          : %0d", cycles);
        $display("Done signal     : %0b", done);

        if (done && checked == 3136 && errors == 0)
            $display("PASS: ALL CONV2 OUTPUTS MATCH");
        else
            $display("FAIL: CONV2 VERIFICATION FAILED");

        $finish;
    end

endmodule
