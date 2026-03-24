import path from "path";
import { fileURLToPath } from "url";
import { Difficulty } from "tja";

const _thisFile = fileURLToPath(import.meta.url);
const _thisDir = path.dirname(_thisFile);
/** Default when no tracks dir is passed: <data>/tracks (sibling of dst/ when compiled). */
export const defaultTracksDir = (): string => path.resolve(_thisDir, "..", "tracks");

export type LabelCliArgs = {
  courseDiff: Difficulty;
  outputDir: string;
  tracksDir: string;
};

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
