import { Course, Difficulty, Song, TJAParser } from "tja";
import fs, { mkdirSync } from "fs";
import path from "path";
import { getCourseNoteTimes } from "./courseNotes.js";
import { fileURLToPath } from "url";

/**
 * Parses and creates JSON files containing timestamp labels for notes in
 * each track.
 *
 * Usage: node labels.js <course difficulty>
 * Supported difficulties are documented here: https://jozsefsallai.github.io/tja-js/classes/Difficulty.html
 *
 * Relative to the script's location; this expects the tracks to be in ../tracks, and
 * places the labels in ../preprocessed/labels/<difficulty>. Label files have the following title naming:
 * Song_Name_Difficulty.json (e.g. Chaoz_Fantasy_Hard.json). Note that if the script finds a song
 * that has been parsed for the same difficulty in the labels folder, it will skip parsing it.
 *
 * If ../preprocessed/labels doesn't exist, the script will create the folder recursively. If
 * ../tracks doesn't exist, an error will be thrown.
 */

// Args
const args: string[] = process.argv.slice(2);
if (args.length != 1) {
  throw Error(
    "Expected 1 argument: course difficulty (easy, normal, hard, etc.)",
  );
}

const diffArg = args[0];
const courseDiff = Difficulty.fromName(diffArg, true);
if (courseDiff === undefined) {
  throw Error(
    `Could not find difficulty '${diffArg}'. Refer to https://jozsefsallai.github.io/tja-js/classes/Difficulty.html for supported difficulties.`,
  );
}

// Constants
// Should be run from the data folder
const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const tracksDir = path.resolve(dirname, "../tracks");
const outDir = path.resolve(
  dirname,
  `../preprocessed/labels/${courseDiff.toString().toLowerCase()}`,
);

// Helpers
// Parses the chart, then returns whether the chart already exists
type ParseChartResult = {
  chartName: string;
  alreadyExists: boolean; // If the chart already existed when the script ran
};

const parseChart = (chart: Song, course: Course): Promise<ParseChartResult> => {
  let title = chart.title.replace(/[\\/:"*?<>| ]+/g, "_").trim();
  const outPath = `${outDir}/${title}_${courseDiff?.toString()}.json`;
  if (fs.existsSync(outPath)) {
    return new Promise((resolve, reject) => {
      resolve({ chartName: title, alreadyExists: true });
    });
  }
  const notes = getCourseNoteTimes(chart, course);

  return new Promise((resolve, reject) => {
    fs.writeFile(outPath, JSON.stringify(notes, null, 2), (err) => {
      if (err) {
        reject("Failed to write to file");
      }
      resolve({ chartName: title, alreadyExists: false });
    });
  });
};

// Main logic
if (!fs.existsSync(tracksDir)) {
  throw Error(
    `Couldn't find tracks directory ${tracksDir}. Check if it exists.`,
  );
}
if (!fs.existsSync(outDir)) {
  console.log(`Creating output directory ${outDir}.`);
  mkdirSync(outDir, { recursive: true });
} else {
  console.log(`Found output directory ${outDir}.`);
}

const songFolders = fs
  .readdirSync(tracksDir, { withFileTypes: true })
  .filter((dirent) => dirent.isDirectory())
  .map((dirent) => dirent.name);

// Find .tja files and parse them
let chartParsedPromises: Promise<ParseChartResult>[] = [];
for (const songFolder of songFolders) {
  const folderPath = `${tracksDir}/${songFolder}`;
  const files = fs.readdirSync(folderPath);
  const tjaFiles = files.filter((file) => file.endsWith(".tja"));

  if (tjaFiles.length !== 1) {
    console.log(`No .tja files/more than 1 .tja file found: ${songFolder}`);
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
  const course = chart.courses.find((c) => courseDiff === c.difficulty);

  if (course) {
    const promise = parseChart(chart, course);
    chartParsedPromises.push(promise);
  } else {
    console.log(
      `Could not find course with desired difficulty in ${tracksDir}/${songFolder}`,
    );
    continue;
  }
}

const results: ParseChartResult[] = await Promise.all(chartParsedPromises);

const parsed = results.reduce((totalParsed, parsedChartRes) => {
  if (parsedChartRes.alreadyExists) {
    console.log(
      `${parsedChartRes.chartName} with difficulty ${courseDiff.toString()} has already been parsed, so it was skipped.`,
    );
  }
  return parsedChartRes.alreadyExists ? totalParsed : totalParsed + 1;
}, 0);

console.log(`Successfully parsed ${parsed} chart(s).`);
