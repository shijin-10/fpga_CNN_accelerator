
module cnn_top_sequential (
    input wire clk,
    input wire rst,
    input wire start,

    input wire signed [7:0] image [0:783],
    input wire signed [7:0] conv1_weights [0:71],
    input wire signed [31:0] conv1_biases [0:7],
    input wire signed [7:0] conv2_weights [0:1151],
    input wire signed [31:0] conv2_biases [0:15],
    input wire signed [7:0] fc_weights [0:7839],
    input wire signed [31:0] fc_biases [0:9],

    output reg busy,
    output reg done,
    output reg [3:0] predicted_digit,
    output reg signed [63:0] scores [0:9]
);

    localparam IDLE       = 4'd0;
    localparam C1_START   = 4'd1;
    localparam C1_WAIT    = 4'd2;
    localparam C2_START   = 4'd3;
    localparam C2_WAIT    = 4'd4;
    localparam FC_START   = 4'd5;
    localparam FC_WAIT    = 4'd6;
    localparam READ_SCORE = 4'd7;
    localparam FIND_MAX   = 4'd8;
    localparam FINISHED   = 4'd9;

    reg [3:0] state;
    reg [2:0] conv1_filter;
    reg [3:0] score_index;

    reg conv1_start;
    reg conv2_start;
    reg fc_start;

    wire conv1_busy, conv1_done, conv1_valid;
    wire [9:0] conv1_pixel_addr;
    wire [3:0] conv1_weight_addr;
    wire signed [31:0] conv1_value;
    wire [9:0] conv1_output_index;

    wire conv2_busy, conv2_done, conv2_valid;
    wire [10:0] conv2_input_addr, conv2_weight_addr;
    wire [3:0] conv2_bias_addr;
    wire [11:0] conv2_output_addr;
    wire signed [31:0] conv2_value;

    wire fc_busy, fc_done;
    wire [9:0] fc_input_addr;
    wire [12:0] fc_weight_addr;
    wire [3:0] fc_bias_addr;
    wire [3:0] fc_read_index;
    wire signed [63:0] fc_read_value;
    wire [3:0] fc_output_index;
    wire signed [63:0] fc_output_value;

    reg signed [31:0] conv1_memory [0:6271];
    reg signed [31:0] conv2_memory [0:3135];

    wire signed [31:0] pool1_memory [0:1567];
    wire signed [7:0] conv2_input_memory [0:1567];

    wire signed [31:0] pool2_memory [0:783];
    wire signed [7:0] fc_input8 [0:783];
    wire signed [31:0] fc_input32 [0:783];

    integer i;
    integer best_index;
    reg signed [63:0] best_score;

    // Conv1 weight and bias selection.
    wire signed [7:0] conv1_weight_data;
    wire signed [31:0] conv1_bias_data;
    wire signed [7:0] conv1_pixel_data;

    assign conv1_pixel_data = image[conv1_pixel_addr];

    assign conv1_weight_data =
        conv1_weights[conv1_filter * 9 + conv1_weight_addr];

    assign conv1_bias_data = conv1_biases[conv1_filter];

    conv1_sequential conv1_unit (
        .clk(clk),
        .rst(rst),
        .start(conv1_start),
        .pixel_data(conv1_pixel_data),
        .weight_data(conv1_weight_data),
        .bias(conv1_bias_data),
        .pixel_address(conv1_pixel_addr),
        .weight_address(conv1_weight_addr),
        .busy(conv1_busy),
        .done(conv1_done),
        .output_valid(conv1_valid),
        .output_index(conv1_output_index),
        .output_value(conv1_value)
    );

    // Conv1 output storage.
    always @(posedge clk) begin
        if (!rst && conv1_valid) begin
            conv1_memory[conv1_filter * 784 + conv1_output_index]
                <= conv1_value;
        end
    end

    // Pooling and requantization after Conv1.
    maxpool_2x2 pool1_unit (
        .input_data(conv1_memory),
        .output_data(pool1_memory)
    );

    requantize_array #(
        .N(1568),
        .MULTIPLIER(64'd2456613),
        .SHIFT(30)
    ) requantize1_unit (
        .input_data(pool1_memory),
        .output_data(conv2_input_memory)
    );

    // Conv2.
    wire signed [7:0] conv2_input_data;
    wire signed [7:0] conv2_weight_data;
    wire signed [31:0] conv2_bias_data;

    assign conv2_input_data = conv2_input_memory[conv2_input_addr];
    assign conv2_weight_data = conv2_weights[conv2_weight_addr];
    assign conv2_bias_data = conv2_biases[conv2_bias_addr];

    conv2_sequential conv2_unit (
        .clk(clk),
        .rst(rst),
        .start(conv2_start),
        .input_data(conv2_input_data),
        .weight_data(conv2_weight_data),
        .bias_data(conv2_bias_data),
        .input_address(conv2_input_addr),
        .weight_address(conv2_weight_addr),
        .bias_address(conv2_bias_addr),
        .output_address(conv2_output_addr),
        .output_value(conv2_value),
        .busy(conv2_busy),
        .done(conv2_done),
        .output_valid(conv2_valid)
    );

    always @(posedge clk) begin
        if (!rst && conv2_valid) begin
            conv2_memory[conv2_output_addr] <= conv2_value;
        end
    end

    // Pooling and requantization after Conv2.
    maxpool_2x2_conv2 pool2_unit (
        .input_data(conv2_memory),
        .output_data(pool2_memory)
    );

    requantize_array #(
        .N(784),
        .MULTIPLIER(64'd2658134),
        .SHIFT(30)
    ) requantize2_unit (
        .input_data(pool2_memory),
        .output_data(fc_input8)
    );

    // Sign-extend INT8 values to the FC module's INT32 input.
    genvar g;
    generate
        for (g = 0; g < 784; g = g + 1) begin : FC_INPUT_EXTEND
            assign fc_input32[g] = {{24{fc_input8[g][7]}}, fc_input8[g]};
        end
    endgenerate

    // FC layer.
    assign fc_read_index = score_index;

    fc_sequential fc_unit (
        .clk(clk),
        .rst(rst),
        .start(fc_start),
        .input_data(fc_input32[fc_input_addr]),
        .weight_data(fc_weights[fc_weight_addr]),
        .bias_data(fc_biases[fc_bias_addr]),
        .input_address(fc_input_addr),
        .weight_address(fc_weight_addr),
        .bias_address(fc_bias_addr),
        .output_read_index(fc_read_index),
        .output_read_value(fc_read_value),
        .output_index(fc_output_index),
        .output_value(fc_output_value),
        .busy(fc_busy),
        .done(fc_done)
    );

    // Main controller.
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            conv1_filter <= 0;
            score_index <= 0;
            conv1_start <= 0;
            conv2_start <= 0;
            fc_start <= 0;
            busy <= 0;
            done <= 0;
            predicted_digit <= 0;
            best_index <= 0;
            best_score <= 0;

            for (i = 0; i < 10; i = i + 1)
                scores[i] <= 0;
        end else begin
            conv1_start <= 0;
            conv2_start <= 0;
            fc_start <= 0;
            done <= 0;

            case (state)
                IDLE: begin
                    busy <= 0;

                    if (start) begin
                        busy <= 1;
                        conv1_filter <= 0;
                        state <= C1_START;
                    end
                end

                C1_START: begin
                    conv1_start <= 1;
                    state <= C1_WAIT;
                end

                C1_WAIT: begin
                    if (conv1_done) begin
                        if (conv1_filter == 7) begin
                            state <= C2_START;
                        end else begin
                            conv1_filter <= conv1_filter + 1'b1;
                            state <= C1_START;
                        end
                    end
                end

                C2_START: begin
                    conv2_start <= 1;
                    state <= C2_WAIT;
                end

                C2_WAIT: begin
                    if (conv2_done)
                        state <= FC_START;
                end

                FC_START: begin
                    fc_start <= 1;
                    state <= FC_WAIT;
                end

                FC_WAIT: begin
                    if (fc_done) begin
                        score_index <= 0;
                        state <= READ_SCORE;
                    end
                end

                READ_SCORE: begin
                    scores[score_index] <= fc_read_value;

                    if (score_index == 9) begin
                        state <= FIND_MAX;
                    end else begin
                        score_index <= score_index + 1'b1;
                    end
                end

                FIND_MAX: begin
                    best_index = 0;
                    best_score = scores[0];

                    for (i = 1; i < 10; i = i + 1) begin
                        if (scores[i] > best_score) begin
                            best_score = scores[i];
                            best_index = i;
                        end
                    end

                    predicted_digit <= best_index[3:0];
                    busy <= 0;
                    done <= 1;
                    state <= FINISHED;
                end

                FINISHED: begin
                    busy <= 0;
                    if (!start)
                        state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                    busy <= 0;
                end
            endcase
        end
    end

endmodule
