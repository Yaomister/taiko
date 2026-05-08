import path from "path";
import { fileURLToPath } from "url";
import { Course, Difficulty, Song } from "tja";
import { getCourseNoteTimes } from "./courseNotes.js";
import fs, { mkdirSync } from "fs";

const _thisFile = fileURLToPath(import.meta.url);
const _thisDir = path.dirname(_thisFile);
/** Default when no tracks dir is passed: <data>/tracks (sibling of dst/ when compiled). */
export const defaultTracksDir = (): string =>
  path.resolve(_thisDir, "..", "tracks");

export type LabelCliArgs = {
  courseDiff: Difficulty;
  outputDir: string;
  tracksDir: string;
};

// Parses the chart, then returns whether the chart already exists
export type ParseChartResult = {
  chartName: string;
  alreadyExists: boolean; // If the chart already existed when the script ran
};

// Throws error if we failt o write to a file
export const createChartJSON = (
  chart: Song,
  course: Course,
  jsonStem: string,
  outDir: string,
): ParseChartResult => {
  const outPath = `${outDir}/${jsonStem}.json`;
  if (fs.existsSync(outPath)) {
    return { chartName: jsonStem, alreadyExists: true };
  }
  const notes = getCourseNoteTimes(chart, course);

  try {
    fs.writeFileSync(outPath, JSON.stringify(notes, null, 2));
  } catch (err) {
    throw Error("Failed to write chart: " + err);
  }

  return { chartName: jsonStem, alreadyExists: false };
};

// Recursively finds all TJA paths in a folder
export const getAllTJAPaths = (dir: string): string[] => {
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
};

export const shortenText = (text: string) =>
  text.length > 50 ? text.substring(0, 50) + "..." : text;

export const validateArgs = (): LabelCliArgs => {
  const args: string[] = process.argv.slice(2);
  if (args.length !== 2 && args.length !== 3) {
    throw Error(
      "Expected 2 or 3 arguments: (1) course difficulty (easy, normal, hard, etc.), (2) output directory (parent folder; a subdirectory named after the difficulty will be created), optionally (3) tracks directory (song folders with .tja). If omitted, defaults to <repo>/data/tracks relative to the compiled script.",
    );
  }

  const diffArg = args[0];
  const outputDirRaw = args[1].trim();
  if (!outputDirRaw) {
    throw Error("Output directory must be a non-empty path.");
  }

  let tracksDir: string;
  if (args.length === 3) {
    const tracksRaw = args[2].trim();
    if (!tracksRaw) {
      throw Error("Tracks directory must be a non-empty path when provided.");
    }
    tracksDir = path.resolve(tracksRaw);
  } else {
    tracksDir = defaultTracksDir();
  }

  const courseDiff = Difficulty.fromName(diffArg, true);
  if (courseDiff === undefined) {
    throw Error(
      `Could not find difficulty '${diffArg}'. Refer to https://jozsefsallai.github.io/tja-js/classes/Difficulty.html for supported difficulties.`,
    );
  }

  return {
    courseDiff,
    outputDir: path.resolve(outputDirRaw),
    tracksDir,
  };
};
