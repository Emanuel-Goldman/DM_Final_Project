import { TextareaProps} from "@cloudscape-design/components";
import { useState } from "react";

// Constraint item type as a tuple: [weight1, weight2, operator]
type ConstraintItem = [number, number, string];

export const useConstraintInput = () => {
    const [constraints, setConstraints] = useState<string>("");
    const [constraintError, setConstraintError] = useState<string | null>(null);

    const handleConstraintsChange: TextareaProps['onChange'] = (event) => {
        setConstraints(event.detail.value); 
        setConstraintError(null);
    };

    /**
     * Parse constraints input string into the format [a1,a2,op],
     * where 'a1' is the coefficient of 'w1', 'a2' is the coefficient of 'w2', and 'op' is the operation between them.
     * Example: '2*w1>=3*w2' --> [2,3,>=]
     */
    const parseConstraints = () => {
        if (constraints == "") {
            return [];
        }
        
        const lines = constraints.split("\n"); // Split input into lines
        const parsed: ConstraintItem[] = [];

        setConstraintError(null);

        for (const line of lines) {
            const regex = /(\d+)\*w(\d+)([<>]=?)(\d+)\*w(\d+)/; // Regex to capture the constraint format
            const match = line.match(regex);

            if (match && match[2] == '1' && match[5] == '2') {
                const weight1 = parseInt(match[1]); // Extract weight coefficient for w1
                const weight2 = parseInt(match[4]); // Extract weight coefficient for w2
                const operator = match[3]; // Extract operator

                const parsedConstraint: ConstraintItem = [weight1, weight2, operator];
                parsed.push(parsedConstraint);
            } else {
                const err = `
                    Invalid constraint format: "${line}".\n 
                    Expected format: "number*w1 operator number*w2" (without spaces). Example: 1*w1>=2*w2.\n
                    Every constraint should appear in a separate line.
                `;
                setConstraintError(err);
                console.error(err);
                return;
            }
        }

        return parsed; 
    };

    return {
        constraints,
        constraintError,
        parseConstraints,
        handleConstraintsChange
    };
};