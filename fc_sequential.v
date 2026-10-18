
module fc_sequential (
    input wire clk,
    input wire rst,
    input wire start,

    input wire signed [31:0] input_data,
    input wire signed [7:0] weight_data,
    input wire signed [31:0] bias_data,

    output reg [9:0] input_address,
    output reg [12:0] weight_address,
    output reg [3:0] bias_address,

    input wire [3:0] output_read_index,
    output wire signed [63:0] output_read_value,

    output reg [3:0] output_index,
    output reg signed [63:0] output_value,

    output reg busy,
    output reg done
);

    localparam IDLE = 2'd0;
    localparam MAC  = 2'd1;
    localparam SAVE = 2'd2;

    reg [1:0] state;

    reg [3:0] class_index;
    reg [9:0] input_index;

    reg signed [63:0] accumulator;
    reg signed [63:0] product;

    reg signed [63:0] score_memory [0:9];

    integer k;

    // Address generation and signed multiplication.
    always @* begin
        input_address = input_index;
        weight_address = class_index * 784 + input_index;
        bias_address = class_index;

        product = $signed(input_data) * $signed(weight_data);
    end

    // Asynchronous read interface for the ten stored scores.
    assign output_read_value = score_memory[output_read_index];

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            class_index <= 0;
            input_index <= 0;
            accumulator <= 0;
            output_index <= 0;
            output_value <= 0;
            busy <= 0;
            done <= 0;

            for (k = 0; k < 10; k = k + 1)
                score_memory[k] <= 0;
        end else begin
            done <= 0;

            case (state)

                IDLE: begin
                    busy <= 0;

                    if (start) begin
                        class_index <= 0;
                        input_index <= 0;
                        accumulator <= 0;
                        busy <= 1;
                        state <= MAC;
                    end
                end

                MAC: begin
                    busy <= 1;

                    if (input_index == 783) begin
                        // Include the final product and bias.
                        score_memory[class_index] <=
                            accumulator +
                            product +
                            {{32{bias_data[31]}}, bias_data};

                        state <= SAVE;
                    end else begin
                        accumulator <= accumulator + product;
                        input_index <= input_index + 1'b1;
                    end
                end

                SAVE: begin
                    if (class_index == 9) begin
                        busy <= 0;
                        done <= 1;

                        output_index <= 0;
                        output_value <= score_memory[0];

                        state <= IDLE;
                    end else begin
                        class_index <= class_index + 1'b1;
                        input_index <= 0;
                        accumulator <= 0;
                        state <= MAC;
                    end
                end

                default: begin
                    state <= IDLE;
                    busy <= 0;
                end

            endcase
        end
    end

endmodule
