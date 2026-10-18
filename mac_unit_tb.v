`timescale 1ns/1ps

module mac_unit_tb;

    reg clk = 0;
    reg rst = 0;
    reg clear = 0;
    reg enable = 0;

    reg signed [7:0] data_in = 0;
    reg signed [7:0] weight = 0;

    wire signed [31:0] accumulator;

    mac_unit dut (
        .clk(clk),
        .rst(rst),
        .clear(clear),
        .enable(enable),
        .data_in(data_in),
        .weight(weight),
        .accumulator(accumulator)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("mac_unit_tb.vcd");
        $dumpvars(0, mac_unit_tb);

        // Reset
        rst = 1;
        @(posedge clk);
        #1 rst = 0;

        // 3 * 4 = 12
        data_in = 3;
        weight = 4;
        enable = 1;
        @(posedge clk);
        #1;
        if (accumulator !== 12)
            $fatal(1, "Test 1 failed: got %0d", accumulator);

        // Add (-2) * 5 = -10; total should be 2
        data_in = -2;
        weight = 5;
        @(posedge clk);
        #1;
        if (accumulator !== 2)
            $fatal(1, "Test 2 failed: got %0d", accumulator);

        // Clear accumulator
        enable = 0;
        clear = 1;
        @(posedge clk);
        #1;
        if (accumulator !== 0)
            $fatal(1, "Clear test failed: got %0d", accumulator);

        clear = 0;

        // Check that disabled MAC holds its value
        data_in = 7;
        weight = 6;
        @(posedge clk);
        #1;
        if (accumulator !== 0)
            $fatal(1, "Enable test failed: got %0d", accumulator);

        $display("PASS: MAC multiplication, accumulation, clear, and enable tests");
        $finish;
    end

endmodule
