<!DOCTYPE html>
<html>
<head>
    <title>text2video Generation</title>
</head>
<body>
    <h1>Video Generation</h1>
    <form method="post" action="/" enctype="multipart/form-data">
        <label for="model">Model:</label>
        <input type="text" id="model" name="model" value="/content/zeroscope_v2_576w" required><br><br>
        <label for="prompt">Prompt:</label>
        <input type="text" id="prompt" name="prompt" required><br><br>
        <label for="negative_prompt">Negative Prompt:</label>
        <input type="text" id="negative_prompt" name="negative_prompt"><br><br>
        <label for="batch_size">Batch Size:</label>
        <input type="number" id="batch_size" name="batch_size" value="1" min="1"><br><br>
        <label for="num_frames">Number of Frames:</label>
        <input type="number" id="num_frames" name="num_frames" value="30" min="1"><br><br>
        <label for="width">Width:</label>
        <input type="number" id="width" name="width" value="256" min="1"><br><br>
        <label for="height">Height:</label>
        <input type="number" id="height" name="height" value="256" min="1"><br><br>
        <label for="num_steps">Number of Steps:</label>
        <input type="number" id="num_steps" name="num_steps" value="25" min="1"><br><br>
        <label for="guidance_scale">Guidance Scale:</label>
        <input type="number" id="guidance_scale" name="guidance_scale" value="23" min="0"><br><br>
        <label for="init_video">Initial Video:</label>
        <input type="file" id="init_video" name="init_video"><br><br>
        <label for="init_weight">Initial Weight:</label>
        <input type="number" id="init_weight" name="init_weight" value="0.5" min="0" max="1" step="0.1"><br><br>
        <label for="fps">Frames Per Second:</label>
        <input type="number" id="fps" name="fps" value="10" min="1"><br><br>
        <label for="device">Device:</label>
        <select id="device" name="device">
            <option value="cuda">cuda</option>
            <option value="cpu">cpu</option>
        </select><br><br>
        <label for="xformers">Use Transformers:</label>
        <input type="checkbox" id="xformers" name="xformers"><br><br>
        <label for="sdp">Use Stochastic Diffusion Process:</label>
        <input type="checkbox" id="sdp" name="sdp"><br><br>
        <label for="lora_path">LoRA Path:</label>
        <input type="text" id="lora_path" name="lora_path"><br><br>
        <label for="lora_rank">LoRA Rank:</label>
        <input type="number" id="lora_rank" name="lora_rank" value="64" min="1"><br><br>
        <label for="remove_watermark">Remove Watermark:</label>
        <input type="checkbox" id="remove_watermark" name="remove_watermark"><br><br>
        <label for="seed">Seed:</label>
        <input type="number" id="seed" name="seed" value="0"><br><br>
        <input type="submit" value="Generate Video">
    </form>
</body>
</html>
