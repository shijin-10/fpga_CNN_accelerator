`timescale 1ns/1ps

module mac_int8_tb;

    reg signed [7:0] a;
    reg signed [7:0] b;
    reg signed [31:0] acc_in;

    wire signed [31:0] acc_out;


    // Instantiate MAC
    mac_int8 uut (
        .a(a),
        .b(b),
        .acc_in(acc_in),
        .acc_out(acc_out)
    );


    initial begin

        // Test 1
        a = 10;
        b = 5;
        acc_in = 0;

        #10;

        $display(
            "Test 1: %d * %d + %d = %d",
            a, b, acc_in, acc_out
        );


        // Test 2
        a = -10;
        b = 5;
        acc_in = 100;

        #10;

        $display(
            "Test 2: %d * %d + %d = %d",
            a, b, acc_in, acc_out
        );


        // Test 3
        a = -20;
        b = -4;
        acc_in = -50;

        #10;

        $display(
            "Test 3: %d * %d + %d = %d",
            a, b, acc_in, acc_out
        );


        // Test 4
        a = 127;
        b = 127;
        acc_in = 0;

        #10;

        $display(
            "Test 4: %d * %d + %d = %d",
            a, b, acc_in, acc_out
        );


        $finish;

    end

endmodule
