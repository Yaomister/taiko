import { Course, Difficulty, Song, TJAParser } from "tja";
import fs from "fs";
import path from "path";
import { getCourseNoteTimes } from "./courseNotes.js";
import { fileURLToPath } from "url";

// Constants
const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const tracksDir = path.resolve(dirname, "../tracks");
const outDir = path.resolve(dirname, "../track_data");
const courseDiff = [Difficulty.Normal, Difficulty.Easy];

// Helpers
const parseChart = (chart: Song, course: Course): void => {
  const notes = getCourseNoteTimes(chart, course);
  let title = chart.title.replace(/[\\/:"*?<>| ]+/g, "_").trim();
  const outPath = `${outDir}/${title}.json`;
  if (fs.existsSync(outPath)) {
    throw new Error(
      `Output file already exists: ${outPath}. Please remove before continuing.`,
    );
  }

  fs.writeFileSync(outPath, JSON.stringify(notes, null, 2), "utf-8");
};

// Main logic
const folders = fs
  .readdirSync(tracksDir, { withFileTypes: true })
  .filter((dirent) => dirent.isDirectory())
  .map((dirent) => dirent.name);

// Find .tja files and parse them
let parsed = 0;
for (const folder of folders) {
  const folderPath = `${tracksDir}/${folder}`;
  const files = fs.readdirSync(folderPath);
  const tjaFiles = files.filter((file) => file.endsWith(".tja"));

  if (tjaFiles.length !== 1) {
    console.log(`No .tja files/more than 1 .tja file found: ${folder}`);
    continue;
  }

  const tjaFilePath = `${folderPath}/${tjaFiles[0]}`;
  const tjaContent = fs.readFileSync(tjaFilePath, "utf-8");
  let chart: Song;
  try {
    chart = TJAParser.parse(tjaContent, true);
  } catch (err) {
    console.log(`Failed to parse ${tjaFilePath}: ${err}`);
    continue;
  }

  //   Find course with appropriate difficulty
  const course = chart.courses.find((c) => courseDiff.includes(c.difficulty));
  if (course) {
    parseChart(chart, course);
  } else {
    console.log(
      `Could not find course with desired difficulty in ${tracksDir}/${folder}`,
    );
  }
  parsed += 1;
}

console.log(`Successfully parsed ${parsed} chart(s).`);
