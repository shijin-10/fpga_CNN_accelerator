module relu_28x28 (
    input signed [31:0] data_in [0:783],
    output reg signed [31:0] data_out [0:783]
);

    integer i;

    always @(*) begin

        for (i = 0; i < 784; i = i + 1) begin

            if (data_in[i] < 0)
                data_out[i] = 0;
            else
                data_out[i] = data_in[i];

        end

    end

endmodule
