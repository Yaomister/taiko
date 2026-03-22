import { Difficulty } from "tja";

export const validateArgs = (): Difficulty => {
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

  return courseDiff;
};
