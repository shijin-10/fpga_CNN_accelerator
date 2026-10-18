module conv2_single (
    input signed [7:0] input_data [0:71],
    input signed [7:0] weights [0:71],
    input signed [31:0] bias,
    output signed [31:0] output_data
);

    integer i;
    integer acc;

    always @(*) begin

        acc = bias;

        for (i = 0; i < 72; i = i + 1) begin
            acc = acc + input_data[i] * weights[i];
        end

    end

    assign output_data = acc;

endmodule
