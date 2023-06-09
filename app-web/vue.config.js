const { defineConfig } = require('@vue/cli-service')
const CopyPlugin = require("copy-webpack-plugin");

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    liveReload: false
  },
  configureWebpack: {
    module: {
      rules: [{ test: /\.onnx$/, use: 'arraybuffer-loader' },
              { test: /\.py$/i, use: 'raw-loader',}],
    },
    plugins: [
      // new CopyPlugin({
        // patterns: [
        //   {
        //     from: './node_modules/onnxruntime-web/dist/ort-wasm.wasm',
        //     to: 'static/',
        //   },             {
        //     from: './node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm',
        //     to: 'static/',
        //   },
          //  {
          //     from: './model',
          //     to: 'static/chunks/pages',
          //   },
          // ],
        // }),
    ]
  }
})
