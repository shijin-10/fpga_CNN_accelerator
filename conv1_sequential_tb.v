
`timescale 1ns/1ps

module conv1_sequential_tb;

    reg clk = 0;
    reg rst = 1;
    reg start = 0;

    reg signed [7:0] pixel_data;
    reg signed [7:0] weight_data;
    reg signed [31:0] bias = 32'sd10;

    wire [9:0] pixel_address;
    wire [3:0] weight_address;
    wire busy, done;
    wire [9:0] output_index;
    wire signed [31:0] output_value;

    reg signed [7:0] image [0:783];
    reg signed [7:0] weights [0:8];

    integer i, r, c, kr, kc;
    integer sr, sc;
    integer expected;
    integer errors;
    integer outputs_checked;
    integer cycles;

    conv1_sequential dut (
        .clk(clk), .rst(rst), .start(start),
        .pixel_data(pixel_data), .weight_data(weight_data),
        .bias(bias), .pixel_address(pixel_address),
        .weight_address(weight_address), .busy(busy),
        .done(done), .output_index(output_index),
        .output_value(output_value)
    );

    always #5 clk = ~clk;

    // Model asynchronous image and weight memories.
    always @* begin
        pixel_data = image[pixel_address];
        weight_data = weights[weight_address];
    end

    initial begin
        errors = 0;
        outputs_checked = 0;

        // Test image and filter contain only ones.
        for (i = 0; i < 784; i = i + 1)
            image[i] = 8'sd1;

        for (i = 0; i < 9; i = i + 1)
            weights[i] = 8'sd1;

        repeat (2) @(negedge clk);
        rst = 0;
        start = 1;
        @(negedge clk);
        start = 0;

        // Check every output as it is saved.
        for (i = 0; i < 784; i = i + 1) begin
            cycles = 0;

            // Wait for the DUT to reach SAVE for this pixel.
            while (dut.state != 2 && cycles < 20) begin
                @(negedge clk);
                cycles = cycles + 1;
            end

            if (dut.state != 2) begin
                $display("ERROR: timeout waiting for pixel %0d", i);
                errors = errors + 1;
            end else begin
                // Independently calculate this pixel's expected value.
                r = i / 28;
                c = i % 28;
                expected = 10;

                for (kr = 0; kr < 3; kr = kr + 1) begin
                    for (kc = 0; kc < 3; kc = kc + 1) begin
                        sr = r + kr - 1;
                        sc = c + kc - 1;

                        if (sr >= 0 && sr < 28 &&
                            sc >= 0 && sc < 28)
                            expected = expected + 1;
                    end
                end

                // SAVE updates the output on the next rising edge.
                @(posedge clk);
                #1;

                if (output_index !== i ||
                    output_value !== expected) begin
                    $display(
                        "FAIL pixel=%0d index=%0d got=%0d expected=%0d",
                        i, output_index, output_value, expected
                    );
                    errors = errors + 1;
                end

                outputs_checked = outputs_checked + 1;

                // Move away from SAVE before checking the next pixel.
                @(negedge clk);
            end
        end

        $display("--------------------------------");
        $display("Outputs checked: %0d", outputs_checked);
        $display("Errors: %0d", errors);

        if (outputs_checked == 784 && errors == 0)
            $display("PASS: ALL 784 CONV1 OUTPUTS MATCH");
        else
            $display("FAIL: CONV1 VERIFICATION INCOMPLETE");

        $finish;
    end

endmodule
