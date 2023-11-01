import { OpenAI } from "langchain/llms/openai";
import { LLMChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import promptTemplate from "./basePrompt.js";
import { InMemoryCache } from "langchain/cache";
import { Redis } from "ioredis";

// Load the Vector Store from the `vectorStore` directory
const store = await HNSWLib.load(
  "vectorStore",
  new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  })
);
console.clear();

const client = new Redis();
const cache = new InMemoryCache();

// OpenAI Configuration
const model = new OpenAI({
  cache,
  temperature: 0.5,
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "text-davinci-002",
  n: 2,
  bestOf: 2,
});

// Parse and initialize the Prompt
const prompt = new PromptTemplate({
  template: promptTemplate,
  inputVariables: ["history", "context", "prompt"],
});

// Create the LLM Chain
const llmChain = new LLMChain({
  llm: model,
  prompt,
});

/**
 * Generates a Response based on history and a prompt.
 * @param {string} history -
 * @param {string} prompt - Th
 */
const generateResponse = async ({ history, prompt }) => {
  // Search for related context/documents in the vectorStore directory
  const cached = await client.get(prompt);
  if (cached) {
    const endTime = new Date().getTime();
    return JSON.parse(cached);
  }
  const data = await store.similaritySearch(prompt, 1);
  const context = [];
  data.forEach((item, i) => {
    context.push(`Context:\n${item.pageContent}`);
  });
  const predictionResult = await llmChain.predict({
    prompt,
    context: context.join("\n\n"),
    history,
  });
  await client.set(prompt, JSON.stringify(predictionResult));
  const endTime2 = new Date().getTime();
  // console.log("Response Time : " + endTime2 - initialTime);

  return predictionResult;
};

export default generateResponse;
