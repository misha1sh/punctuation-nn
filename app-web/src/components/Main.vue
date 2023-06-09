<template>
  <div class="hello">
    <h1>Исходный текст: </h1>
    <n-input
      type="textarea"
      v-model:value="input"
      placeholder="Исходный текст:"
      clearable
      :autosize="{
        minRows: 5,
        maxRows: 20
      }"
      style="font-size:1.2em"
    />
    <n-space align="center">
      <n-button
        type="info"
        @click="handleInput"
        :disabled="showLoadingBar">Раставить пунктуацию</n-button>
        <n-button
      type="info"
        @click="clearPunctuation"
        :disabled="input.length == 0">
        Очистить пунктуацию из исходного текста</n-button>

    <n-checkbox v-model:checked="show_diff" :disabled="results_diff.length == 0 || showLoadingBar">
      Показывать разницу с исходным текстом
    </n-checkbox>

    </n-space>
    <h1>Результаты: </h1>

    <n-spin :show="showLoadingBar">
      <n-card>
        <div v-if="show_diff" style="font-size:1.2em">
          <div v-for="[index, part] in results_diff.entries()" :key="index" style="display: inline;">
            <div v-if="part.added" style="background-color:#8aff8a; display: inline; white-space: pre-wrap">{{part.value}}</div>
            <div v-if="part.removed" style="background-color:#ffcfcf; display: inline; white-space: pre-wrap">{{part.value}}</div>
            <div v-if="!part.added && !part.removed" style="display: inline; white-space: pre-wrap">{{part.value}}</div>
          </div>
        </div>
        <div v-else style="font-size:1.2em; white-space: pre-wrap">{{results}}</div>
      </n-card>

      <template #description>
        Загрузка
      </template>
    </n-spin>
  </div>
</template>

<script setup>
  import { NCard, NInput, NButton, NSpin, NSpace, NCheckbox } from 'naive-ui'
</script>

<script>
      // #<!-- @input="handleInput" -->
import {ref} from 'vue'

import { InferenceSession, env, Tensor } from 'onnxruntime-web';

env.wasm.wasmPaths = {
  'ort-wasm.wasm': "/js/ort-wasm.wasm",
  'ort-wasm-threaded.wasm':  "/js/ort-wasm-threaded.wasm",
  'ort-wasm-simd.wasm':  "/js/ort-wasm-simd.wasm",
  'ort-wasm-simd-threaded.wasm':  "/js/ort-wasm-simd-threaded.wasm",
}
// const model = require("../assets/model.onnx")
// import buffer from "../assets/model2.onnx";
// console.log(buffer)

const session_promise =
    fetch('https://storage.yandexcloud.net/misha-sh-objects/model2.onnx')
        .then(response => response.arrayBuffer())
        .then(buffer => {
            return InferenceSession.create(buffer,
                { executionProviders: ['wasm'] });
        })

import { diffChars } from 'diff'

import { loadPyodide } from 'pyodide'
const pyodide_promise = loadPyodide({
    indexURL: "/pyodide/",
  });


import code from '../assets/code.py'
export default {
  name: 'HelloWorld',
  props: {
  },
  data() {
    return {
      input: "",
      results: "",
      results_diff: [],
      showLoadingBar: ref(true),
      show_diff: ref(true)
    }
  },
  created() {
    console.log("CREATED")
    async function run() {
      console.log("RUN")
      const pyodide = await pyodide_promise;
      await pyodide.loadPackage("micropip")
      const session = await session_promise;

      async function infer(float32_array) {
        const buf = float32_array.getBuffer("f32")

        const tensor = new Tensor('float32', buf.data, [1 , 32, 489]);
        const prom = session.run({input: tensor});
        const output = (await prom).output;
        return Array.from(output.data)
      }

      await pyodide.registerJsModule("jsinfer", {
        "infer": infer
      })
      // console.log(await pyodide.runPythonAsync(code))
      pyodide.globals.set('text', 'Тест.')
      console.log(await pyodide.runPythonAsync(code + "\nawait infer_optimal(params, text)"))
    }
    run().then(() => {
      this.showLoadingBar = false;
    })
    pyodide_promise.then(pyodide => {
      console.log(pyodide)

    })
  },
  methods: {
    async handleInput() {
      this.showLoadingBar = true;

      const inp = this.input;

      const pyodide = await pyodide_promise;
      pyodide.globals.set('text', inp)
      const res = await pyodide.runPythonAsync(code + "\nawait infer_optimal(params, text)")

      this.results = res;
      this.results_diff = diffChars(inp, res);

      this.showLoadingBar = false;
    },
    clearPunctuation() {
      this.input = this.input.replaceAll('.', '').replaceAll(',', '');
    },
  }
}
</script>

<style scoped>

</style>
