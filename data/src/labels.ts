import { Course, Difficulty, Song, TJAParser } from "tja";
import fs, { mkdirSync } from "fs";
import path from "path";
import { getCourseNoteTimes } from "./courseNotes.js";
import {
  createChartJSON,
  getAllTJAPaths as getAllTJAsRecursively,
  ParseChartResult,
  shortenText,
  validateArgs,
} from "./labelUtils.js";

/**
 * Parses and creates JSON files containing timestamp and type labels for all notes in each track.
 * The script searches the tracks directory recursively, but the track folders must have a .tja
 * file and an audio track.
 *
 * Labels go in <output directory>/<difficulty>/. Each JSON file is named
 *  after the parent folder of the .tja (sanitized for the filesystem),  e.g. .../Chaoz Fantasy/chart.tja -> Chaoz_Fantasy.json.
 *
 * Usage: node labels.js <course difficulty> <output directory> [tracks directory]
 * Supported difficulties are documented here: https://jozsefsallai.github.io/tja-js/classes/Difficulty.html
 *
 * Behaviour:
 * - If tracks directory is omitted, it defaults to ../tracks next to the compiled script (<data>/tracks).
 *
 *
 * - If the script finds a song that has been parsed for the same difficulty in the labels folder, it will skip parsing it.
 *
 * - If <output directory>/<difficulty> doesn't exist, the script will create it recursively. If
 *   the tracks directory doesn't exist, an error will be thrown.
 */

// Args
const { courseDiff, outputDir: labelsParentDir, tracksDir } = validateArgs();
const outDir = path.join(labelsParentDir, courseDiff.toString().toLowerCase());

if (!fs.existsSync(tracksDir)) {
  throw Error(
    `Couldn't find tracks directory ${tracksDir}. Check if it exists.`,
  );
}
if (!fs.existsSync(outDir)) {
  mkdirSync(outDir, { recursive: true });
} else {
  console.log(`Found output directory ${outDir}.`);
}

const filePaths: string[] = getAllTJAsRecursively(tracksDir);
// For error message at the end
const failedJSONWrite: string[] = [];
const failedParse: string[] = [];
const missingDiff: string[] = [];

// Find .tja files and parse them
let results: ParseChartResult[] = [];
for (const filePath of filePaths) {
  const content = fs.readFileSync(filePath, { encoding: "utf-8" });
  let chart: Song;

  try {
    chart = TJAParser.parse(content, true);
  } catch (err) {
    const name = path.basename(filePath);
    failedParse.push(name);
    continue;
  }

  //   Find course with appropriate difficulty
  const course = chart.courses.find((c) => courseDiff === c.difficulty);

  if (course) {
    const songFolder = path.dirname(filePath);
    const jsonStem = path.basename(songFolder);
    let res;
    try {
      res = createChartJSON(chart, course, jsonStem, outDir);
    } catch (err) {
      failedJSONWrite.push(chart.title);
      continue;
    }
    results.push(res);
  } else {
    missingDiff.push(chart.title);
    continue;
  }
}

const newlyParsed = results.reduce(
  (prev, curr) => (curr.alreadyExists ? prev : prev + 1),
  0,
);

if (failedJSONWrite.length !== 0) {
  const songNames = shortenText(failedJSONWrite.join(", "));
  console.log(
    `Failed to create label files for ${failedJSONWrite.length} song(s): ` +
      songNames,
  );
}
if (failedParse.length !== 0) {
  const songNames = shortenText(failedParse.join(", "));
  console.log(
    `Invalid TJA formatting found for ${failedParse.length} song(s): ` +
      songNames,
  );
}
if (missingDiff.length !== 0) {
  const songNames = shortenText(missingDiff.join(", "));
  console.log(
    `Couldn't find requested difficulty for ${missingDiff.length} song(s): ` +
      songNames,
  );
}

if (newlyParsed === 0) {
  console.log("No new labels created, all songs found were already parsed.");
} else {
  console.log(`Successfully created labels for ${newlyParsed} chart(s).`);
}
