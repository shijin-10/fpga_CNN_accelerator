
`timescale 1ns/1ps

module fc_sequential_tb;

    reg clk = 0;
    reg rst = 1;
    reg start = 0;

    reg signed [31:0] input_memory [0:783];
    reg signed [7:0]  weight_memory [0:7839];
    reg signed [31:0] bias_memory [0:9];

    reg signed [63:0] reference [0:9];

    wire signed [31:0] input_data;
    wire signed [7:0] weight_data;
    wire signed [31:0] bias_data;

    wire [9:0] input_address;
    wire [12:0] weight_address;
    wire [3:0] bias_address;

    reg [3:0] output_read_index;
    wire signed [63:0] output_read_value;

    wire [3:0] output_index;
    wire signed [63:0] output_value;

    wire busy;
    wire done;

    integer i;
    integer errors;
    integer cycles;
    integer ref_file;
    integer scan_result;

    assign input_data = input_memory[input_address];
    assign weight_data = weight_memory[weight_address];
    assign bias_data = bias_memory[bias_address];

    fc_sequential dut (
        .clk(clk),
        .rst(rst),
        .start(start),

        .input_data(input_data),
        .weight_data(weight_data),
        .bias_data(bias_data),

        .input_address(input_address),
        .weight_address(weight_address),
        .bias_address(bias_address),

        .output_read_index(output_read_index),
        .output_read_value(output_read_value),

        .output_index(output_index),
        .output_value(output_value),

        .busy(busy),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        $display("Loading FC input, weights, and biases...");

        $readmemh("fc_input.mem", input_memory);
        $readmemh("fc_weights.mem", weight_memory);
        $readmemh("fc_bias_int32.mem", bias_memory);

        // Reference file is decimal text, written by fc_layer_tb.v.
        ref_file = $fopen("fc_output.mem", "r");

        if (ref_file == 0) begin
            $display("ERROR: Cannot open fc_output.mem");
            $finish;
        end

        for (i = 0; i < 10; i = i + 1) begin
            scan_result = $fscanf(ref_file, "%d", reference[i]);

            if (scan_result != 1) begin
                $display("ERROR: Cannot read reference score %0d", i);
                $fclose(ref_file);
                $finish;
            end
        end

        $fclose(ref_file);

        repeat (2) @(negedge clk);
        rst = 0;

        start = 1;
        @(negedge clk);
        start = 0;

        cycles = 0;

        while (!done && cycles < 10000) begin
            @(negedge clk);
            cycles = cycles + 1;
        end

        if (!done) begin
            $display("FAIL: FC timed out.");
            $finish;
        end

        // Read and compare all ten scores.
        errors = 0;

        $display("----------------------------------------");

        for (i = 0; i < 10; i = i + 1) begin
            output_read_index = i;
            #1;

            $display(
                "Digit %0d: sequential=%0d reference=%0d",
                i, output_read_value, reference[i]
            );

            if (output_read_value !== reference[i])
                errors = errors + 1;
        end

        $display("----------------------------------------");
        $display("Scores checked : 10");
        $display("Errors         : %0d", errors);
        $display("Cycles         : %0d", cycles);

        if (errors == 0)
            $display("PASS: ALL FC SCORES MATCH");
        else
            $display("FAIL: FC VERIFICATION FAILED");

        $finish;
    end

endmodule
