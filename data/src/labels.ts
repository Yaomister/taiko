import { Course, Difficulty, Song, TJAParser } from "tja";
import fs, { mkdirSync } from "fs";
import path from "path";
import { getCourseNoteTimes } from "./courseNotes.js";
import { validateArgs } from "./labelUtils.js";

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

// function sanitizeLabelJsonStem(folderName: string): string {
//   return folderName.replace(/[\\/:"*?<>| ]+/g, "_").trim();
// }

// Helpers
// Parses the chart, then returns whether the chart already exists
type ParseChartResult = {
  chartName: string;
  alreadyExists: boolean; // If the chart already existed when the script ran
};

const parseChart = (
  chart: Song,
  course: Course,
  jsonStem: string,
): Promise<ParseChartResult> => {
  const outPath = `${outDir}/${jsonStem}.json`;
  if (fs.existsSync(outPath)) {
    return new Promise((resolve, reject) => {
      resolve({ chartName: jsonStem, alreadyExists: true });
    });
  }
  const notes = getCourseNoteTimes(chart, course);

  return new Promise((resolve, reject) => {
    fs.writeFile(outPath, JSON.stringify(notes, null, 2), (err) => {
      if (err) {
        reject("Failed to write to file");
      }
      resolve({ chartName: jsonStem, alreadyExists: false });
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
  mkdirSync(outDir, { recursive: true });
} else {
  console.log(`Found output directory ${outDir}.`);
}

// Recursively find all TJA paths
function getAllTJAPaths(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  return entries.flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      return getAllTJAPaths(fullPath);
    } else if (entry.isFile() && entry.name.endsWith(".tja")) {
      return [fullPath];
    }

    return [];
  });
}

const filePaths: string[] = getAllTJAPaths(tracksDir);

// Find .tja files and parse them
let chartParsedPromises: Promise<ParseChartResult>[] = [];
for (const filePath of filePaths) {
  const tjaContent = fs.readFileSync(filePath, "utf-8");
  let chart: Song;
  try {
    chart = TJAParser.parse(tjaContent, true);
  } catch (err) {
    console.log(`Failed to parse ${filePath}: ${err}`);
    continue;
  }

  //   Find course with appropriate difficulty
  const course = chart.courses.find((c) => courseDiff === c.difficulty);

  if (course) {
    const songFolder = path.dirname(filePath);
    const jsonStem = path.basename(songFolder);
    const promise = parseChart(chart, course, jsonStem);
    chartParsedPromises.push(promise);
  } else {
    console.log(`Could not find course with desired difficulty in ${filePath}`);
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
