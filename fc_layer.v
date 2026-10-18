
module fc_layer (
    input signed [31:0] input_data [0:783],
    input signed [7:0]  weights [0:7839],
    input signed [31:0] biases [0:9],
    output reg signed [63:0] output_data [0:9]
);

    integer n, i;
    integer weight_index;

    reg signed [63:0] accumulator;
    reg signed [63:0] input_extended;
    reg signed [63:0] weight_extended;
    reg signed [63:0] product;

    always @(*) begin
        for (n = 0; n < 10; n = n + 1) begin
            accumulator = {{32{biases[n][31]}}, biases[n]};

            for (i = 0; i < 784; i = i + 1) begin
                weight_index = n * 784 + i;

                input_extended =
                    {{32{input_data[i][31]}}, input_data[i]};

                weight_extended =
                    {{56{weights[weight_index][7]}},
                      weights[weight_index]};

                product = input_extended * weight_extended;
                accumulator = accumulator + product;
            end

            output_data[n] = accumulator;
        end
    end

endmodule
