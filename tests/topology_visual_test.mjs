import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const [html, css, js] = await Promise.all([
  readFile(new URL("../index.html", import.meta.url), "utf8"),
  readFile(new URL("../styles.css", import.meta.url), "utf8"),
  readFile(new URL("../app.js", import.meta.url), "utf8"),
]);

test("self-evolution uses a labeled canvas topology instead of DOM line segments", () => {
  assert.match(html, /<canvas[\s\S]*?id="topology-canvas"/);
  assert.match(html, /Conceptual topology/);
  assert.match(html, /Retained/);
  assert.match(html, /Added/);
  assert.match(html, /Pruned/);
  assert.doesNotMatch(css, /\.mini-edge\b/);
  assert.doesNotMatch(css, /\.mini-node\b/);
});

test("public page does not show an ICLR 2027 venue marker", () => {
  assert.doesNotMatch(html, /ICLR 2027/i);
});

test("each evolution mode declares a distinct graph and meaningful node labels", () => {
  for (const mode of ["baseline", "expert", "repair", "scratch"]) {
    assert.match(js, new RegExp(`${mode}: \\{[\\s\\S]*?graph:`));
  }
  for (const label of ["START", "READ", "VERIFY", "RESPOND", "END"]) {
    assert.match(js, new RegExp(`label: "${label}"`));
  }
  assert.match(js, /status: "added"/);
  assert.match(js, /\["infer", "respond", "pruned"\]/);
});
